param(
    [string]$Config = "configs/ppi_graph.yaml"
)

$ErrorActionPreference = "Stop"

python -m goai_graph.build_graph --config $Config
python -m goai_graph.train --config $Config --variant no_graph
python -m goai_graph.train --config $Config --variant real_ppi
python -m goai_graph.train --config $Config --variant rewired_ppi
