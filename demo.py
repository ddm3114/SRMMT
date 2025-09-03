#!/usr/bin/env python3
"""
SRMMT Demo - Sentencing Reasoning Model for Mock Trial
A basic demonstration of a sentencing reasoning system for mock trials.
"""

import json
import random
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class CaseEvidence:
    """Evidence presented in a mock trial case"""
    evidence_type: str
    description: str
    severity_impact: float  # -1.0 to 1.0, negative mitigating, positive aggravating

@dataclass
class DefendantProfile:
    """Defendant information"""
    age: int
    prior_convictions: int
    employment_status: str
    family_circumstances: str

@dataclass
class Case:
    """Mock trial case"""
    case_id: str
    charge: str
    defendant: DefendantProfile
    evidence: List[CaseEvidence]
    guilty_verdict: bool

class SentencingReasoningModel:
    """Basic sentencing reasoning model for mock trials"""
    
    def __init__(self):
        self.base_sentences = {
            "theft": {"min": 6, "max": 24, "base": 12},  # months
            "assault": {"min": 12, "max": 36, "base": 18},
            "fraud": {"min": 18, "max": 48, "base": 24},
            "burglary": {"min": 24, "max": 60, "base": 36},
        }
        
        self.mitigating_factors = {
            "first_offense": -0.2,
            "employed": -0.1,
            "young_age": -0.15,
            "family_support": -0.1,
        }
        
        self.aggravating_factors = {
            "repeat_offender": 0.3,
            "unemployed": 0.1,
            "victim_impact": 0.2,
        }
    
    def calculate_sentence(self, case: Case) -> Dict[str, Any]:
        """Calculate recommended sentence based on case details"""
        if not case.guilty_verdict:
            return {
                "verdict": "NOT GUILTY",
                "sentence_months": 0,
                "reasoning": ["Defendant found not guilty - no sentence applied"]
            }
        
        charge_lower = case.charge.lower()
        base_info = None
        for charge_type, info in self.base_sentences.items():
            if charge_type in charge_lower:
                base_info = info
                break
        
        if not base_info:
            base_info = {"min": 6, "max": 24, "base": 12}  # default
        
        base_sentence = base_info["base"]
        adjustment_factor = 0.0
        reasoning = [f"Base sentence for {case.charge}: {base_sentence} months"]
        
        # Analyze defendant factors
        if case.defendant.prior_convictions == 0:
            adjustment_factor += self.mitigating_factors["first_offense"]
            reasoning.append("First offense: -20% sentence reduction")
        elif case.defendant.prior_convictions >= 2:
            adjustment_factor += self.aggravating_factors["repeat_offender"]
            reasoning.append(f"Repeat offender ({case.defendant.prior_convictions} priors): +30% sentence increase")
        
        if case.defendant.age < 25:
            adjustment_factor += self.mitigating_factors["young_age"]
            reasoning.append("Young defendant (under 25): -15% sentence reduction")
        
        if case.defendant.employment_status.lower() == "employed":
            adjustment_factor += self.mitigating_factors["employed"]
            reasoning.append("Defendant employed: -10% sentence reduction")
        elif case.defendant.employment_status.lower() == "unemployed":
            adjustment_factor += self.aggravating_factors["unemployed"]
            reasoning.append("Defendant unemployed: +10% sentence increase")
        
        # Analyze evidence
        evidence_impact = sum(e.severity_impact for e in case.evidence)
        if evidence_impact > 0:
            adjustment_factor += min(evidence_impact * 0.1, 0.3)  # cap at 30%
            reasoning.append(f"Aggravating evidence impact: +{min(evidence_impact * 10, 30):.1f}% sentence increase")
        elif evidence_impact < 0:
            adjustment_factor += max(evidence_impact * 0.1, -0.3)  # cap at 30%
            reasoning.append(f"Mitigating evidence impact: {max(evidence_impact * 10, -30):.1f}% sentence reduction")
        
        # Calculate final sentence
        final_sentence = base_sentence * (1 + adjustment_factor)
        final_sentence = max(base_info["min"], min(base_info["max"], final_sentence))
        final_sentence = round(final_sentence)
        
        reasoning.append(f"Total adjustment factor: {adjustment_factor:+.1%}")
        reasoning.append(f"Final sentence: {final_sentence} months")
        
        return {
            "verdict": "GUILTY",
            "sentence_months": final_sentence,
            "base_sentence": base_sentence,
            "adjustment_factor": adjustment_factor,
            "reasoning": reasoning
        }

def create_sample_cases() -> List[Case]:
    """Create sample mock trial cases for demonstration"""
    cases = [
        Case(
            case_id="CASE001",
            charge="Theft",
            defendant=DefendantProfile(
                age=22,
                prior_convictions=0,
                employment_status="employed",
                family_circumstances="lives with parents"
            ),
            evidence=[
                CaseEvidence("witness_testimony", "Store clerk identified defendant", 0.3),
                CaseEvidence("security_footage", "Clear video of theft", 0.4),
                CaseEvidence("character_witness", "Employer testified to good character", -0.2),
            ],
            guilty_verdict=True
        ),
        Case(
            case_id="CASE002",
            charge="Assault",
            defendant=DefendantProfile(
                age=35,
                prior_convictions=3,
                employment_status="unemployed",
                family_circumstances="divorced, no children"
            ),
            evidence=[
                CaseEvidence("medical_records", "Victim hospitalized for 3 days", 0.6),
                CaseEvidence("police_report", "Defendant intoxicated at scene", 0.3),
                CaseEvidence("witness_testimony", "Unprovoked attack", 0.4),
            ],
            guilty_verdict=True
        ),
        Case(
            case_id="CASE003",
            charge="Fraud",
            defendant=DefendantProfile(
                age=28,
                prior_convictions=1,
                employment_status="employed",
                family_circumstances="married with two children"
            ),
            evidence=[
                CaseEvidence("financial_records", "Falsified documents found", 0.5),
                CaseEvidence("victim_impact", "Elderly victim lost life savings", 0.7),
                CaseEvidence("cooperation", "Defendant cooperated with investigation", -0.3),
            ],
            guilty_verdict=True
        ),
        Case(
            case_id="CASE004",
            charge="Theft",
            defendant=DefendantProfile(
                age=19,
                prior_convictions=0,
                employment_status="student",
                family_circumstances="lives with guardian"
            ),
            evidence=[
                CaseEvidence("alibi", "Solid alibi provided by multiple witnesses", -0.8),
                CaseEvidence("mistaken_identity", "Similar appearance to actual perpetrator", -0.6),
            ],
            guilty_verdict=False
        ),
    ]
    return cases

def run_demo():
    """Run the SRMMT demo"""
    print("=" * 60)
    print("SRMMT - Sentencing Reasoning Model for Mock Trial")
    print("=" * 60)
    print()
    
    model = SentencingReasoningModel()
    cases = create_sample_cases()
    
    for i, case in enumerate(cases, 1):
        print(f"CASE {i}: {case.case_id}")
        print(f"Charge: {case.charge}")
        print(f"Defendant: Age {case.defendant.age}, {case.defendant.prior_convictions} prior convictions")
        print(f"Employment: {case.defendant.employment_status}")
        print(f"Family: {case.defendant.family_circumstances}")
        print()
        
        print("Evidence:")
        for evidence in case.evidence:
            impact = "aggravating" if evidence.severity_impact > 0 else "mitigating" if evidence.severity_impact < 0 else "neutral"
            print(f"  - {evidence.description} ({impact})")
        print()
        
        result = model.calculate_sentence(case)
        
        print(f"VERDICT: {result['verdict']}")
        if result['verdict'] == 'GUILTY':
            print(f"SENTENCE: {result['sentence_months']} months")
            print()
            print("REASONING:")
            for reason in result['reasoning']:
                print(f"  • {reason}")
        else:
            print("REASONING:")
            for reason in result['reasoning']:
                print(f"  • {reason}")
        
        print("\n" + "-" * 60 + "\n")
    
    print("Demo completed!")
    print("\nNote: This is a simplified demonstration model.")
    print("Real sentencing involves complex legal considerations not captured here.")

if __name__ == "__main__":
    run_demo()
