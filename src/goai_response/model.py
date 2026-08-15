"""A factorised background + response + calibration predictor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch
from torch import nn


LOCKED_EXPERT_SCALES = frozenset({0.0, 0.25, 0.5, 0.75, 1.0})


@dataclass
class ResponseComponentOutput:
    """Named decomposition used by M7 diagnostics without changing legacy tuples."""

    absolute: torch.Tensor
    background_plus_calibration: torch.Tensor
    response: torch.Tensor
    background: torch.Tensor
    calibration: torch.Tensor
    background_universal: torch.Tensor
    background_strain: torch.Tensor
    response_universal: torch.Tensor
    response_strain: torch.Tensor
    response_chemical: torch.Tensor
    response_pair: torch.Tensor
    response_prior: torch.Tensor
    strain_gate: torch.Tensor
    chemical_gate: torch.Tensor
    pair_gate: torch.Tensor


class ResponseDecompositionRegressor(nn.Module):
    """Keeps biological response and observation calibration separate.

    All outputs are in per-protein standardised log2 units.  The response head
    is low rank so that 4k proteins share statistical strength without assuming
    that a static PPI edge is an active causal edge in every condition.
    """

    def __init__(
        self,
        response_input_dim: int,
        background_input_dim: int,
        observation_input_dim: int,
        n_proteins: int,
        hidden_dim: int,
        response_rank: int,
        calibration_rank: int,
        dropout: float,
        calibration_enabled: bool = True,
        response_basis: str = "learned",
        cell_input_dim: int = 0,
        perturbation_input_dim: int = 0,
        interaction_mode: str = "independent_legacy",
        calibration_plate_start: int = -1,
        calibration_plate_end: int = -1,
        calibration_plate_dropout: float = 0.0,
        response_prior_learnable_scale: bool = False,
        general_cell_input_dim: int = 0,
        general_perturbation_input_dim: int = 0,
        n_strain_entities: int = 0,
        n_chemical_entities: int = 0,
        n_pair_entities: int = 0,
        background_strain_expert_enabled: bool = True,
        response_strain_expert_enabled: bool = True,
        response_chemical_expert_enabled: bool = True,
        response_pair_expert_enabled: bool = False,
        strain_expert_scale: float = 1.0,
        chemical_expert_scale: float = 1.0,
        pair_expert_scale: float = 1.0,
        entity_dropout: float = 0.0,
        allow_research_expert_scale_override: bool = False,
    ) -> None:
        super().__init__()
        if response_basis not in {"learned", "fixed_svd"}:
            raise ValueError("response_basis must be learned or fixed_svd")
        if interaction_mode not in {"independent_legacy", "shared_concat", "shared_gate", "shared_film", "shared_general_experts"}:
            raise ValueError("unsupported interaction mode")
        if not 0.0 <= entity_dropout < 1.0:
            raise ValueError("entity_dropout must be in [0, 1)")
        if min(strain_expert_scale, chemical_expert_scale, pair_expert_scale) < 0.0:
            raise ValueError("expert scales cannot be negative")
        scales = {float(strain_expert_scale), float(chemical_expert_scale), float(pair_expert_scale)}
        if not allow_research_expert_scale_override and not scales <= LOCKED_EXPERT_SCALES:
            raise ValueError(
                "expert scales must be in the locked grid "
                "{0, 0.25, 0.5, 0.75, 1}; research override must be explicit"
            )
        self.interaction_mode = interaction_mode
        if interaction_mode == "independent_legacy":
            self.response_encoder = nn.Sequential(nn.Linear(response_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, response_rank))
            self.background_encoder = nn.Sequential(nn.Linear(background_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_proteins))
        elif interaction_mode == "shared_general_experts":
            if general_cell_input_dim <= 0:
                raise ValueError("shared_general_experts requires general cell inputs")
            if min(n_strain_entities, n_chemical_entities) <= 0:
                raise ValueError("shared_general_experts requires non-empty entity vocabularies")
            self.general_cell_encoder = nn.Sequential(
                nn.Linear(general_cell_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
            )
            self.general_background_decoder = nn.Linear(hidden_dim, n_proteins)
            self.general_response_encoder = nn.Sequential(
                nn.Linear(hidden_dim + general_perturbation_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, response_rank),
            )
            self.background_strain_expert_enabled = background_strain_expert_enabled
            self.response_strain_expert_enabled = response_strain_expert_enabled
            self.response_chemical_expert_enabled = response_chemical_expert_enabled
            self.response_pair_expert_enabled = response_pair_expert_enabled
            self.n_strain_entities = n_strain_entities
            self.n_chemical_entities = n_chemical_entities
            self.n_pair_entities = n_pair_entities
            if background_strain_expert_enabled:
                self.background_strain_embeddings = nn.Embedding(
                    n_strain_entities + 1, hidden_dim, padding_idx=0
                )
                self.background_strain_decoder = nn.Linear(hidden_dim, n_proteins, bias=False)
                nn.init.zeros_(self.background_strain_embeddings.weight)
            if response_strain_expert_enabled:
                self.response_strain_embeddings = nn.Embedding(
                    n_strain_entities + 1, response_rank, padding_idx=0
                )
                nn.init.zeros_(self.response_strain_embeddings.weight)
            if response_chemical_expert_enabled:
                self.response_chemical_embeddings = nn.Embedding(
                    n_chemical_entities + 1, response_rank, padding_idx=0
                )
                nn.init.zeros_(self.response_chemical_embeddings.weight)
            if response_pair_expert_enabled:
                if n_pair_entities <= 0:
                    raise ValueError("pair expert requires a non-empty fold-fit pair vocabulary")
                self.response_pair_embeddings = nn.Embedding(
                    n_pair_entities + 1,
                    response_rank,
                    padding_idx=0,
                )
                self.response_pair_context = nn.Linear(
                    hidden_dim, response_rank, bias=False
                )
                nn.init.zeros_(self.response_pair_embeddings.weight)
                # Begin as the original static pair expert.  Context modulation
                # is learned only when pair evidence supports it.
                nn.init.zeros_(self.response_pair_context.weight)
        else:
            if cell_input_dim <= 0 or perturbation_input_dim <= 0:
                raise ValueError("shared interaction modes require cell and perturbation inputs")
            self.cell_encoder = nn.Sequential(nn.Linear(cell_input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
            self.background_decoder = nn.Linear(hidden_dim, n_proteins)
            self.response_encoder = nn.Sequential(
                nn.Linear(hidden_dim + perturbation_input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, response_rank),
            )
            if interaction_mode == "shared_gate":
                self.cell_modulator = nn.Sequential(nn.Linear(perturbation_input_dim, hidden_dim), nn.Sigmoid())
            elif interaction_mode == "shared_film":
                self.cell_modulator = nn.Linear(perturbation_input_dim, hidden_dim * 2)
        self.response_basis = response_basis
        self.register_buffer("response_center", torch.zeros(n_proteins))
        if response_basis == "learned":
            self.response_proteins = nn.Parameter(torch.randn(response_rank, n_proteins) * 0.02)
        else:
            self.register_buffer("response_proteins", torch.zeros(response_rank, n_proteins))
        self.calibration_enabled = calibration_enabled
        self.calibration_plate_start = calibration_plate_start
        self.calibration_plate_end = calibration_plate_end
        self.calibration_plate_dropout = calibration_plate_dropout
        self.register_buffer(
            "calibration_input_center", torch.zeros(observation_input_dim)
        )
        self.strain_expert_scale = float(strain_expert_scale)
        self.chemical_expert_scale = float(chemical_expert_scale)
        self.pair_expert_scale = float(pair_expert_scale)
        self.entity_dropout = float(entity_dropout)
        self.training_stage = "joint"
        if calibration_enabled:
            self.calibration_encoder = nn.Linear(observation_input_dim, calibration_rank, bias=False)
            self.calibration_proteins = nn.Parameter(torch.zeros(calibration_rank, n_proteins))
        if response_prior_learnable_scale:
            self.response_prior_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer("response_prior_scale", torch.tensor(1.0))

    def set_fixed_response_basis(self, center: torch.Tensor, basis: torch.Tensor) -> None:
        """Install a fold-fitted response basis without making it trainable."""
        if self.response_basis != "fixed_svd":
            raise ValueError("set_fixed_response_basis requires response_basis='fixed_svd'")
        if center.shape != self.response_center.shape:
            raise ValueError(f"center must have shape {tuple(self.response_center.shape)}")
        if basis.shape != self.response_proteins.shape:
            raise ValueError(f"basis must have shape {tuple(self.response_proteins.shape)}")
        with torch.no_grad():
            self.response_center.copy_(center.to(self.response_center))
            self.response_proteins.copy_(basis.to(self.response_proteins))

    def set_calibration_input_center(self, center: torch.Tensor) -> None:
        """Install the fold-fit observation mean used for hard centering."""
        if center.shape != self.calibration_input_center.shape:
            raise ValueError(
                f"calibration center must have shape {tuple(self.calibration_input_center.shape)}"
            )
        if not torch.isfinite(center).all():
            raise ValueError("calibration center must be finite")
        with torch.no_grad():
            self.calibration_input_center.copy_(center.to(self.calibration_input_center))

    @staticmethod
    def _entity_vector(values: torch.Tensor | None, batch: int, label: str) -> torch.Tensor:
        if values is None:
            raise ValueError(f"shared_general_experts requires {label}")
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]
        if values.ndim != 1 or len(values) != batch:
            raise ValueError(f"{label} must have shape [batch] or [batch, 1]")
        return values.long()

    @staticmethod
    def _entity_gate(values: torch.Tensor | None, batch: int, label: str) -> torch.Tensor:
        if values is None:
            raise ValueError(f"shared_general_experts requires {label}")
        if values.ndim == 1:
            values = values.unsqueeze(1)
        if values.shape != (batch, 1):
            raise ValueError(f"{label} must have shape [batch, 1]")
        return values

    @staticmethod
    def balanced_entity_regime_gates(
        strain_gate: torch.Tensor,
        chemical_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask a mini-batch into near-equal 00/10/01/11 support regimes."""
        if strain_gate.shape != chemical_gate.shape or strain_gate.ndim != 2 or strain_gate.shape[1] != 1:
            raise ValueError("entity gates must have matching shape [batch, 1]")
        patterns = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=strain_gate.dtype,
            device=strain_gate.device,
        )
        assignments = patterns[
            torch.arange(len(strain_gate), device=strain_gate.device) % len(patterns)
        ]
        assignments = assignments[torch.randperm(len(assignments), device=strain_gate.device)]
        return (
            strain_gate * assignments[:, 0:1],
            chemical_gate * assignments[:, 1:2],
        )

    def set_training_stage(self, stage: str) -> None:
        """Freeze non-target components for deterministic M7 warmup stages."""
        if stage not in {"universal", "strain", "chemical", "pair", "joint"}:
            raise ValueError(f"unsupported training stage: {stage}")
        if self.interaction_mode != "shared_general_experts" and stage != "joint":
            raise ValueError("expert warmup stages require shared_general_experts")
        prefixes = {
            "strain": ("background_strain_", "response_strain_"),
            "chemical": ("response_chemical_",),
            "pair": ("response_pair_",),
        }
        expert_prefixes = tuple(prefix for values in prefixes.values() for prefix in values)
        for name, parameter in self.named_parameters():
            if stage == "joint":
                trainable = True
            elif stage == "universal":
                trainable = not name.startswith(expert_prefixes)
            else:
                trainable = name.startswith(prefixes[stage])
            parameter.requires_grad_(trainable)
        self.training_stage = stage

    @staticmethod
    def _is_expert_state(name: str) -> bool:
        """Return whether a tensor belongs to a fold-fit residual expert."""
        return name.startswith(
            (
                "background_strain_",
                "response_strain_",
                "response_chemical_",
                "response_pair_",
            )
        )

    def universal_state_sha256(self) -> str:
        """Hash every non-expert tensor in a device-independent byte contract.

        This receipt lets two independently fitted OOF producers prove that
        they started their residual-expert stages from the exact same universal
        trunk, decoder, and calibration state.
        """
        digest = hashlib.sha256()
        for name, tensor in sorted(self.state_dict().items()):
            if self._is_expert_state(name):
                continue
            value = tensor.detach().cpu().contiguous()
            header = json.dumps(
                [name, str(value.dtype), list(value.shape)],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "little"))
            digest.update(raw)
        return digest.hexdigest()

    def project_response_prior(self, response_prior: torch.Tensor) -> torch.Tensor:
        """Project a full-protein target-stat prior through the shared basis.

        Prototypes remain a fold-local research prior, not an entity expert or
        a new M7 model number.  This projection ensures their delivered
        response still lies in the exact decoder space used by R_U/R_s/R_c/R_sc.
        """

        if response_prior.ndim != 2 or response_prior.shape[1] != self.response_center.numel():
            raise ValueError("response prior shape does not match protein decoder")
        basis = self.response_proteins
        gram = basis @ basis.transpose(0, 1)
        coefficients = response_prior @ basis.transpose(0, 1) @ torch.linalg.pinv(gram)
        return coefficients @ basis

    def copy_universal_state_from(
        self, source: "ResponseDecompositionRegressor"
    ) -> str:
        """Copy only common tensors from a universal-only pretraining model.

        Expert tensors remain at their explicit zero/residual initialization.
        Shape and key equality are enforced so an architecture mismatch cannot
        silently turn a purported fold-matched warm start into a partial load.
        """
        source_state = {
            name: value
            for name, value in source.state_dict().items()
            if not self._is_expert_state(name)
        }
        target_state = self.state_dict()
        target_common = {
            name: value
            for name, value in target_state.items()
            if not self._is_expert_state(name)
        }
        if set(source_state) != set(target_common):
            missing = sorted(set(target_common).difference(source_state))
            unexpected = sorted(set(source_state).difference(target_common))
            raise ValueError(
                "Universal warm-start state contract differs: "
                f"missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            for name, target in target_common.items():
                source_value = source_state[name]
                if source_value.shape != target.shape or source_value.dtype != target.dtype:
                    raise ValueError(
                        f"Universal warm-start tensor mismatch for {name}: "
                        f"source={tuple(source_value.shape)}/{source_value.dtype}, "
                        f"target={tuple(target.shape)}/{target.dtype}"
                    )
                target.copy_(source_value.to(device=target.device))
        source_hash = source.universal_state_sha256()
        if self.universal_state_sha256() != source_hash:
            raise RuntimeError("Universal warm-start state changed during copy")
        return source_hash

    def _forward_named(
        self,
        response_inputs: torch.Tensor,
        background_inputs: torch.Tensor,
        observation_inputs: torch.Tensor,
        treatment: torch.Tensor,
        cell_inputs: torch.Tensor | None = None,
        perturbation_inputs: torch.Tensor | None = None,
        response_prior: torch.Tensor | None = None,
        general_cell_inputs: torch.Tensor | None = None,
        general_perturbation_inputs: torch.Tensor | None = None,
        strain_indices: torch.Tensor | None = None,
        chemical_indices: torch.Tensor | None = None,
        strain_seen: torch.Tensor | None = None,
        chemical_seen: torch.Tensor | None = None,
        pair_indices: torch.Tensor | None = None,
        pair_seen: torch.Tensor | None = None,
    ) -> ResponseComponentOutput:
        if treatment.ndim != 2 or treatment.shape[1] != 1:
            raise ValueError("treatment must have shape [batch, 1]")
        zero_output = torch.zeros(
            (len(treatment), self.response_center.shape[0]),
            dtype=treatment.dtype,
            device=treatment.device,
        )
        background_strain = zero_output
        response_strain = zero_output
        response_chemical = zero_output
        response_pair = zero_output
        zero_gate = torch.zeros((len(treatment), 1), dtype=treatment.dtype, device=treatment.device)
        used_strain_gate = zero_gate
        used_chemical_gate = zero_gate
        used_pair_gate = zero_gate
        if self.interaction_mode == "independent_legacy":
            background = self.background_encoder(background_inputs)
            response_latent = self.response_encoder(response_inputs)
            background_universal = background
            response_universal = self.response_center + response_latent @ self.response_proteins
        elif self.interaction_mode == "shared_general_experts":
            if general_cell_inputs is None or general_perturbation_inputs is None:
                raise ValueError("shared_general_experts requires general semantic tensors")
            strain_index = self._entity_vector(strain_indices, len(treatment), "strain_indices")
            chemical_index = self._entity_vector(chemical_indices, len(treatment), "chemical_indices")
            pair_index = self._entity_vector(pair_indices, len(treatment), "pair_indices")
            raw_strain_gate = self._entity_gate(strain_seen, len(treatment), "strain_seen").to(treatment)
            raw_chemical_gate = self._entity_gate(chemical_seen, len(treatment), "chemical_seen").to(treatment)
            raw_pair_gate = self._entity_gate(pair_seen, len(treatment), "pair_seen").to(treatment)
            used_strain_gate, used_chemical_gate = raw_strain_gate, raw_chemical_gate
            used_pair_gate = raw_pair_gate
            if self.training:
                if self.training_stage == "universal":
                    used_strain_gate, used_chemical_gate, used_pair_gate = zero_gate, zero_gate, zero_gate
                elif self.training_stage == "strain":
                    used_chemical_gate, used_pair_gate = zero_gate, zero_gate
                elif self.training_stage == "chemical":
                    used_strain_gate, used_pair_gate = zero_gate, zero_gate
                elif self.training_stage == "pair":
                    # R_sc is the final residual expert.  Its frozen stage must
                    # therefore see the already fitted R_s/R_c components (and
                    # B_s for the absolute loss) exactly as inference will use
                    # them.  Hiding those gates here would train R_sc against
                    # R_U alone and then double-add entity effects at inference.
                    used_strain_gate = raw_strain_gate
                    used_chemical_gate = raw_chemical_gate
                elif self.entity_dropout > 0.0:
                    used_strain_gate, used_chemical_gate = self.balanced_entity_regime_gates(
                        raw_strain_gate, raw_chemical_gate
                    )
                    used_pair_gate = raw_pair_gate * used_strain_gate * used_chemical_gate
            h_cell = self.general_cell_encoder(general_cell_inputs)
            background_universal = self.general_background_decoder(h_cell)
            if self.background_strain_expert_enabled:
                background_strain = used_strain_gate * self.background_strain_decoder(
                    self.background_strain_embeddings(strain_index)
                )
            background = background_universal + self.strain_expert_scale * background_strain
            universal_latent = self.general_response_encoder(
                torch.cat((h_cell, general_perturbation_inputs), dim=1)
            )
            response_universal = self.response_center + universal_latent @ self.response_proteins
            if self.response_strain_expert_enabled:
                response_strain = used_strain_gate * (
                    self.response_strain_embeddings(strain_index) @ self.response_proteins
                )
            if self.response_chemical_expert_enabled:
                response_chemical = used_chemical_gate * (
                    self.response_chemical_embeddings(chemical_index) @ self.response_proteins
                )
            if self.response_pair_expert_enabled:
                pair_index = torch.where(used_pair_gate[:, 0] > 0, pair_index, torch.zeros_like(pair_index))
                pair_latent = self.response_pair_embeddings(pair_index)
                pair_latent = pair_latent * (
                    1.0 + torch.tanh(self.response_pair_context(h_cell))
                )
                response_pair = used_pair_gate * (pair_latent @ self.response_proteins)
            response = (
                response_universal
                + self.strain_expert_scale * response_strain
                + self.chemical_expert_scale * response_chemical
                + self.pair_expert_scale * response_pair
            )
        else:
            if cell_inputs is None or perturbation_inputs is None:
                raise ValueError("shared interaction mode requires cell and perturbation tensors")
            h_cell = self.cell_encoder(cell_inputs)
            background = self.background_decoder(h_cell)
            if self.interaction_mode == "shared_gate":
                h_cell = h_cell * self.cell_modulator(perturbation_inputs)
            elif self.interaction_mode == "shared_film":
                scale, shift = self.cell_modulator(perturbation_inputs).chunk(2, dim=1)
                h_cell = h_cell * (1.0 + torch.tanh(scale)) + shift
            response_latent = self.response_encoder(torch.cat((h_cell, perturbation_inputs), dim=1))
            response_universal = self.response_center + response_latent @ self.response_proteins
            response = response_universal
            background_universal = background
        if self.interaction_mode in {"independent_legacy", "shared_concat", "shared_gate", "shared_film"}:
            response = response_universal
        prior_component = zero_output
        if response_prior is not None and response_prior.shape[1] > 0:
            if response_prior.shape != response.shape:
                raise ValueError("response prior shape does not match response output")
            if self.interaction_mode == "shared_general_experts":
                prior_value = self.project_response_prior(response_prior)
            else:
                # Exact legacy equation for historical M2/M6 checkpoints.
                prior_value = response_prior
            prior_component = self.response_prior_scale * prior_value
            response = response + prior_component
        calibrated_inputs = observation_inputs - self.calibration_input_center
        if (
            self.training
            and self.calibration_plate_dropout > 0
            and self.calibration_plate_start >= 0
            and self.calibration_plate_end > self.calibration_plate_start
        ):
            # Drop only the centered plate deviation.  Because its fold-fit
            # expectation is zero, masking preserves a zero expected input;
            # masking raw one-hot values would shift the calibration mean.
            calibrated_inputs = calibrated_inputs.clone()
            keep = (torch.rand((len(observation_inputs), 1), device=observation_inputs.device) >= self.calibration_plate_dropout).to(observation_inputs.dtype)
            calibrated_inputs[:, self.calibration_plate_start:self.calibration_plate_end] *= keep
        calibration = (
            self.calibration_encoder(calibrated_inputs) @ self.calibration_proteins
            if self.calibration_enabled
            else torch.zeros_like(background)
        )
        absolute = background + calibration + treatment * response
        return ResponseComponentOutput(
            absolute=absolute,
            background_plus_calibration=background + calibration,
            response=response,
            background=background,
            calibration=calibration,
            background_universal=background_universal,
            background_strain=self.strain_expert_scale * background_strain,
            response_universal=response_universal,
            response_strain=self.strain_expert_scale * response_strain,
            response_chemical=self.chemical_expert_scale * response_chemical,
            response_pair=self.pair_expert_scale * response_pair,
            response_prior=prior_component,
            strain_gate=used_strain_gate,
            chemical_gate=used_chemical_gate,
            pair_gate=used_pair_gate,
        )

    def forward_named_components(self, *args, **kwargs) -> ResponseComponentOutput:
        """Return the full named decomposition, including all gated M7 experts."""
        return self._forward_named(*args, **kwargs)

    def forward_components(
        self,
        response_inputs: torch.Tensor,
        background_inputs: torch.Tensor,
        observation_inputs: torch.Tensor,
        treatment: torch.Tensor,
        cell_inputs: torch.Tensor | None = None,
        perturbation_inputs: torch.Tensor | None = None,
        response_prior: torch.Tensor | None = None,
        general_cell_inputs: torch.Tensor | None = None,
        general_perturbation_inputs: torch.Tensor | None = None,
        strain_indices: torch.Tensor | None = None,
        chemical_indices: torch.Tensor | None = None,
        strain_seen: torch.Tensor | None = None,
        chemical_seen: torch.Tensor | None = None,
        pair_indices: torch.Tensor | None = None,
        pair_seen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self._forward_named(
            response_inputs, background_inputs, observation_inputs, treatment,
            cell_inputs, perturbation_inputs, response_prior,
            general_cell_inputs, general_perturbation_inputs,
            strain_indices, chemical_indices, strain_seen, chemical_seen,
            pair_indices, pair_seen,
        )
        return (
            result.absolute,
            result.background_plus_calibration,
            result.response,
            result.background,
            result.calibration,
        )

    def forward(
        self,
        response_inputs: torch.Tensor,
        background_inputs: torch.Tensor,
        observation_inputs: torch.Tensor,
        treatment: torch.Tensor,
        cell_inputs: torch.Tensor | None = None,
        perturbation_inputs: torch.Tensor | None = None,
        response_prior: torch.Tensor | None = None,
        general_cell_inputs: torch.Tensor | None = None,
        general_perturbation_inputs: torch.Tensor | None = None,
        strain_indices: torch.Tensor | None = None,
        chemical_indices: torch.Tensor | None = None,
        strain_seen: torch.Tensor | None = None,
        chemical_seen: torch.Tensor | None = None,
        pair_indices: torch.Tensor | None = None,
        pair_seen: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        absolute, background_plus_calibration, response, _, _ = self.forward_components(
            response_inputs,
            background_inputs,
            observation_inputs,
            treatment,
            cell_inputs,
            perturbation_inputs,
            response_prior,
            general_cell_inputs,
            general_perturbation_inputs,
            strain_indices,
            chemical_indices,
            strain_seen,
            chemical_seen,
            pair_indices,
            pair_seen,
        )
        return absolute, background_plus_calibration, response
