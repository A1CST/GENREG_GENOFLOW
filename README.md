CrashOut: OLA, GENREG, and OLM Models
Status: Archived - No further updates
This repository contains three years of independent AI research exploring evolutionary learning as an alternative to gradient descent. All models, code, and documentation are released under MIT license for anyone interested in exploring these ideas further.
What's Here
This repo contains implementations of several evolutionary learning architectures:

OLA (Organic Learning Architecture) - Core evolutionary learning system using trust-based selection
GENREG (Genetic Regulation) - Language modeling via evolutionary dynamics instead of backpropagation
OLM (Organic Learning Model) - Multi-modal system with biological-inspired drives and homeostatic control
O-CLIP - Zero-shot image classification using evolutionary learning
Snake/Gym Agents - Reinforcement learning via evolution

Core Concept
Traditional neural networks learn via gradient descent and backpropagation. These systems replace that entirely with:

Variation - Random mutations to network weights
Selection - Trust-based culling of poor performers
Memory - Population diversity maintenance

The goal was continuous learning without catastrophic forgetting, using compact models (~15MB) that learn through mutation and selection rather than gradient updates.
Key Components
GENREG (Genetic Regulation)
Located in /GENREG_TIERED_LEARNING/
Language modeling system that achieved 93%+ next-word prediction accuracy through evolutionary selection rather than backpropagation. Uses curriculum learning with progressive tier advancement.
Key files:

train.py - Main training loop with evolutionary selection
genome.py - LSTM-based genome architecture with protein primitives
environment.py - Tiered curriculum environments

Warning: Achieved high prediction accuracy but embeddings did not organize semantically (0.8% category clustering vs 50% random baseline). The LSTM layers learned patterns, but not through meaningful embedding representations.
OLA (Organic Learning Architecture)
Located in /Isolated_OLA_Base/
Foundational evolutionary learning system. Core principles:

Trust accumulation through performance
Gentle culling (top performers survive)
Diversity maintenance via population dynamics
Environmental pressure scaling

Successfully applied to Snake gameplay (95.8% success rate over 500K episodes) and other tasks.
Critical insight: Culling rate matters more than trust thresholds. Aggressive culling (keeping top 10%) works better than trust-based filtering.
O-CLIP (Zero-Shot Classification)
Located in /CLIP/
Evolutionary approach to CLIP-style vision-language models. Achieved 20-23% validation accuracy on ImageNet-1K zero-shot classification through curriculum learning from binary → 101-way classification.
GYM Environments
Located in /GYM/
Reinforcement learning agents trained via evolution for OpenAI Gym tasks. Includes Snake gameplay achieving sustained performance without catastrophic forgetting.
Known limitation: Agents learned specific control mappings rather than generalizable strategies. Performance collapsed when controls were inverted, indicating memorization rather than true continuous learning.
OLM (Organic Learning Model)
Located in /genome_1/
Earlier multi-modal architecture with:

Multi-stage LSTM pipelines
VAE sensory processing
Homeostatic control mechanisms
Biological-inspired behavioral drives (hunger, fear, curiosity)

Evolved into the simpler OLA/GENREG systems after distilling core mechanisms.
Documentation
/Documentation/ contains ~60 research documents covering:

Architectural decisions and iterations
Training results and failure analyses
Theoretical frameworks
Experimental logs

Key Findings
What Worked

Evolutionary selection can achieve competitive performance on specific tasks
Trust-based population dynamics prevent catastrophic forgetting in constrained domains
Compact models (15MB) can match larger networks on narrow tasks
Curriculum learning with tier progression enables scaling

What Didn't Work

Semantic organization: GENREG embeddings remained random despite prediction accuracy
Transfer learning: Agents failed when task conditions changed (control inversion)
Continuous learning: Not truly continual - more like task-specific memorization
Generalization: Performance tied to specific training distributions

Critical Lessons

Prediction accuracy ≠ understanding - High task performance doesn't guarantee meaningful representations
Evolutionary pressure optimizes for survival, not semantics - Selection finds shortcuts, not general intelligence
Catastrophic forgetting avoided ≠ continuous learning - Maintaining performance isn't the same as learning new skills
Scaling isn't solved - These approaches don't address fundamental scaling challenges differently than gradient descent

Performance Data
GENREG Language Model:

Training accuracy: 93%+ on next-word prediction
Embedding semantic clustering: 0.8% (effectively random)
Generalization: Unable to verify due to evaluation bugs

Snake Agents:

Peak performance: 95.8% success rate (25+ food consumption)
Sustained: 500K+ episodes without catastrophic forgetting
Control inversion: Complete collapse (learned control mappings, not strategy)

O-CLIP:

ImageNet-1K zero-shot: 20-23% accuracy via curriculum learning
Binary classification: 85%+ through gentle culling

Why This Is Archived
After three years of development:

Core promise of "continuous learning" proved to be task-specific memorization
High performance metrics masked lack of semantic understanding
Transfer learning failed basic generalization tests
Evolutionary approaches didn't solve fundamental AI challenges differently

The work is released as-is for anyone interested in exploring these ideas, learning from the failures, or building on specific components that showed promise.
Usage Notes
Models are provided as checkpoints (.pt, .pth, .pkl files). Code is research-quality - expect bugs, inconsistent naming, and minimal documentation. This was exploratory work, not production software.
If you use this work:

Understand the limitations documented above
Don't assume high task accuracy means the system "understands" anything
Test generalization rigorously - specific task performance can be misleading
Consider this a case study in what doesn't work as much as what does

License
MIT License - Use freely, no warranties provided.
Contact
Repository owner is no longer actively researching AI. This code is provided as-is with no support or updates planned.
