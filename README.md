# Aegis MICA
Multiple Input Curious Agent

Single-node, Protopost-powered version of Aegis Graph

## TODO
- Allow setting size by env var
- Allow specifying initial connections via env var, which are probed for their sizes
    - Set up node so other nodes can probe this node's size before it resolves the sizes of its own links (to avoid deadlock waiting for node sizes)
- Web interface for configuration
    - add links
    - remove links
    - adjust nice
- Avoid creating link models (etc) if we are just going to load them immediately after
- API for being controllable via a graph web UI that can see multiple Aegis nodes
- Annealing links in/out of system
- Tensorboard logging?
    - log actions
    - log rnd training loss
    - log scaled and unscaled curiousity
    - log agent metrics?