# Distributed Systems

## Consensus

The Raft consensus algorithm elects a single leader that accepts client writes
and replicates a log to follower nodes. If the leader fails, followers hold an
election for a new term. Raft is designed to be easier to understand than Paxos.

## Container orchestration

Kubernetes orchestrates containers using pods (the smallest deployable unit),
deployments (declarative rollouts), and services (stable network endpoints).

## Caching

A cache stores the results of expensive operations for fast reuse. Redis is an
in-memory data store commonly used as a cache and message broker.
