# Attention Serve

## Update

- 2025/05/31: Initial version.

## Description

This is a distributed version of the attention server. It consist of two groups of workers
- MLP workers: process each batch one by one, and then turn around to the attention workers.
- Attention workers: process the attention.

