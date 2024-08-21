## README 

# MPPI controller for quadrotor carrying a payload with a switching dynamics 

This repository contains the implementation of a Model Predictive Path Integral (MPPI) controller for a quadrotor carrying a payload. The controller handles dynamic switching between slack and taut conditions of the cable during transportation. The implementation is heavily based on the dynamics and control concepts described in the RotorTM paper.

## Table of contents


- Item Introduction
- Item Installation
- Item    Usage
- Item    Theory
    - Subitem  MPPI Overview
    - Subitem  Dynamic Switching Mechanism
- Item Code Structure
- Item Contributions

## Introduction

The project simulates a quadrotor carrying a payload using a cable, which introduces hybrid dynamics. The MPPI controller is designed to manage the nonlinear and hybrid nature of the system, dynamically switching between slack and taut cable conditions during the payload's transportation. The core simulation code is implemented in quadrotor2.py.

## Installation

For the installation we send the reader to https://github.com/tombelv/sbmpc.git which our MPPI library is based on

## Usage

In order to execute the simulation the user has to run
 
```python

run quadrotor2.py
```


## Theory

### MPPI Overview

Model Predictive Path Integral (MPPI) control is a stochastic optimal control method that uses importance sampling to estimate the control policy. MPPI is particularly well-suited for systems with complex dynamics and constraints, such as a quadrotor with a cable-suspended payload. It leverages a large number of samples from possible future states to determine the optimal control action at each timestep.

In the context of this project, MPPI is used to compute the control inputs for the quadrotor, ensuring that it follows a desired trajectory while managing the slack and taut conditions of the cable.

### Dynamic Switching Mechanism

The dynamic switching between slack and taut cable conditions is a critical aspect of the control strategy. The RotorTM framework provides a model for the transient hybrid dynamics, accounting for the collisions and velocity transitions that occur when the cable's state changes.

In quadrotor2.py, the functions func_slack and func_taut define the system's behavior in slack and taut conditions, respectively. The controller switches between these states based on the cable's length and the relative velocities of the quadrotor and payload, ensuring accurate and stable control.
For more theoretical details, refer to the RotorTM paper​:  

paper: https://arxiv.org/abs/2205.05140


## Code Structure

- quadrotor2.py: Core simulation file that defines the dynamics, control logic, and state update mechanisms.
- simulation.py: Containains the simulation settings
- solver.py: Contains the MPPI library. 
- model.py: This file contains different aspects of the model rangig from number of inputs, degree of freedom, velocities to different methods to integratation


    
