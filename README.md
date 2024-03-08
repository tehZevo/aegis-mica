# Aegis MICA
Multiple Input Curious Agent

Single-node, Protopost-powered version of Aegis Graph

## TODO
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