# Phase 9 Central Execution Log: Residual Interaction, Sensitivity & Adaptation

**Branch:** `research/phase9-residual-and-decoder-adaptation`  
**Start Date:** November 7, 2026  
**Hardware:** AMD Radeon RX 9070 XT (ROCm 7.2.4)  
**Primary Objective:** Systematically determine whether the 51.7% non-linear interaction residual is predictable interaction structure, irreducible conditional ambiguity, decoder sensitivity anisotropy, or an autoencoder interface bottleneck.

---

## Log Protocol
- Every test run is recorded here with:
  1. Timestamp & Hardware state
  2. Exact mathematical target & loss formulation
  3. Sample count, split, and baseline definitions
  4. Numerical metrics (standardized MSE, NMSE, audio SDR vs. mixture and ceiling)
  5. Any anomalies, unexpected behaviors, or results requiring scrutiny
  6. Git commit hash of corresponding script and artifacts

---

## Stage A: Diagnosing the Residual

*Status: In Progress...*
