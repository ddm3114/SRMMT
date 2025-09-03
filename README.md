# SRMMT
Sentencing Reasoning Model for Mock Trial

## Overview
SRMMT is a demonstration project that simulates a sentencing reasoning model for mock trial scenarios. This educational tool helps understand how various factors might influence sentencing decisions in a simplified legal context.

## Features
- **Case Analysis**: Process mock trial cases with defendant profiles and evidence
- **Sentencing Logic**: Apply rule-based reasoning to calculate recommended sentences
- **Transparency**: Provide clear reasoning for sentencing decisions
- **Demo Cases**: Includes sample cases for demonstration purposes

## Quick Start

### Run the Demo
```bash
python3 demo.py
```

### Sample Output
The demo processes several mock trial cases and shows:
- Case details (charge, defendant profile, evidence)
- Verdict determination
- Sentence calculation with reasoning
- Factors that influenced the decision

## Project Structure
```
.
├── README.md          # This file
├── demo.py           # Main demo script
└── requirements.txt  # Dependencies (currently none for basic demo)
```

## How It Works
1. **Case Input**: Define cases with charges, defendant profiles, and evidence
2. **Factor Analysis**: Evaluate mitigating and aggravating factors
3. **Sentence Calculation**: Apply adjustments to base sentences
4. **Reasoning Output**: Provide transparent explanation of the decision

## Educational Purpose
This project is designed for educational purposes to demonstrate:
- Rule-based decision making systems
- Legal reasoning simulation
- Transparent AI decision processes
- Mock trial scenario handling

## Limitations
- Simplified model for demonstration purposes
- Does not represent actual legal sentencing guidelines
- Not suitable for real legal decision making
- Based on fictional scenarios

## Future Enhancements
- Machine learning integration
- Web interface
- More sophisticated reasoning models
- Integration with legal databases
- Natural language processing for case descriptions
