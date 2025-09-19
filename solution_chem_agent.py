#!/usr/bin/env python3
"""
Molecular Property Analysis Agent using RDKit

Functions:
- get_molecules_by_name(query: str, limit: int = 5) -> list
- smiles_to_mol(smiles: str) -> object
- compute_properties(mol: object) -> dict
- find_candidates_for_property(property_name: str, target_value: float, query: str, tolerance: float = 0.5, limit: int = 10) -> list

CLI example:
python solution_chem_agent.py --query "benzodiazepine" --property logp --target 2.0 --tolerance 0.5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    from rdkit import RDLogger

    # Reduce RDKit warning noise
    RDLogger.DisableLog('rdApp.*')
except Exception as e:  # pragma: no cover - handled at runtime
    Chem = None  # type: ignore
    Crippen = None  # type: ignore
    Descriptors = None  # type: ignore
    Lipinski = None  # type: ignore
    rdMolDescriptors = None  # type: ignore
    _rdkit_import_error = e
else:
    _rdkit_import_error = None


DEFAULT_DATA_PATH = "/app/resources/test_molecules.json"

SUPPORTED_PROPERTIES = {
    "logp": "Partition coefficient (LogP)",
    "mw": "Molecular Weight",
    "donors": "Number of hydrogen bond donors",
    "acceptors": "Number of hydrogen bond acceptors",
    "rotatable_bonds": "Number of rotatable bonds",
    "tpsa": "Topological Polar Surface Area",
}


class ChemAgentError(Exception):
    pass


def _ensure_rdkit_available() -> None:
    if _rdkit_import_error is not None:
        raise ChemAgentError(
            f"RDKit is required but could not be imported: {_rdkit_import_error}"
        )


def _load_dataset(path: str = DEFAULT_DATA_PATH) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Molecule dataset not found at '{path}'. Ensure the test data file exists."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ChemAgentError(f"Invalid JSON in dataset file: {e}") from e

    if not isinstance(data, list):
        raise ChemAgentError("Dataset must be a JSON array of molecule records.")

    # Normalize records to have at least name and smiles
    normalized: List[Dict[str, Any]] = []
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        name = rec.get("name") or rec.get("Name")
        smiles = rec.get("smiles") or rec.get("SMILES") or rec.get("smile")
        synonyms = rec.get("synonyms") or rec.get("Synonyms") or []
        if name and smiles:
            nrec = {"name": str(name), "smiles": str(smiles)}
            if isinstance(synonyms, list):
                nrec["synonyms"] = [str(s) for s in synonyms]
            normalized.append(nrec)
    return normalized


def smiles_to_mol(smiles: str) -> Optional[object]:
    """Convert a SMILES string to an RDKit Mol. Returns None if invalid."""
    _ensure_rdkit_available()
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        mol = None
    return mol


def compute_properties(mol: object) -> Dict[str, Any]:
    """Compute key molecular properties for the given RDKit Mol.

    Returns a dict with keys: logp, mw, donors, acceptors, rotatable_bonds, tpsa
    """
    _ensure_rdkit_available()
    if mol is None:
        raise ValueError("mol must be a valid RDKit Mol, got None")
    # Some RDKit functions expect a Mol object specifically
    if not hasattr(mol, "GetNumAtoms"):
        raise ValueError("mol does not appear to be a valid RDKit Mol")

    try:
        logp = float(Crippen.MolLogP(mol))
        mw = float(Descriptors.MolWt(mol))
        donors = int(Lipinski.NumHDonors(mol))
        acceptors = int(Lipinski.NumHAcceptors(mol))
        rot = int(Lipinski.NumRotatableBonds(mol))
        tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception as e:
        raise ChemAgentError(f"Failed to compute properties: {e}") from e

    return {
        "logp": logp,
        "mw": mw,
        "donors": donors,
        "acceptors": acceptors,
        "rotatable_bonds": rot,
        "tpsa": tpsa,
    }


def get_molecules_by_name(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Load molecules from dataset and filter by name or substring (case-insensitive).

    Returns a list of records: { name, smiles, properties }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    dataset = _load_dataset()
    q = query.strip().lower()

    matches: List[Dict[str, Any]] = []
    for rec in dataset:
        name = rec.get("name", "")
        smiles = rec.get("smiles", "")
        syns = rec.get("synonyms", [])
        searchable = [name] + ([s for s in syns if isinstance(s, str)] if isinstance(syns, list) else [])
        if any(q in (s or "").lower() for s in searchable):
            mol = smiles_to_mol(smiles)
            if mol is None:
                continue
            props = compute_properties(mol)
            matches.append({
                "name": name,
                "smiles": smiles,
                "properties": props,
            })
            if len(matches) >= limit:
                break

    return matches


def _validate_property_name(property_name: str) -> str:
    if not isinstance(property_name, str) or not property_name.strip():
        raise ValueError("property_name must be a non-empty string")
    key = property_name.strip().lower()
    if key not in SUPPORTED_PROPERTIES:
        raise ValueError(
            f"Unsupported property '{property_name}'. Supported: {', '.join(SUPPORTED_PROPERTIES.keys())}"
        )
    return key


def _score_match(value: float, target: float, tolerance: float) -> float:
    diff = abs(value - target)
    if tolerance <= 0:
        return 1.0 if diff == 0 else 0.0
    # 1.0 at exact match, 0.0 at tolerance boundary
    score = max(0.0, 1.0 - (diff / tolerance))
    return float(score)


def find_candidates_for_property(
    property_name: str,
    target_value: float,
    query: str,
    tolerance: float = 0.5,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Find molecules with property close to target.

    Returns a list of dicts with fields:
      - name, smiles, properties
      - property_name, target_value, value, difference, match_quality
      - justification
    """
    key = _validate_property_name(property_name)

    if not isinstance(target_value, (int, float)) or math.isnan(float(target_value)):
        raise ValueError("target_value must be a valid number")
    target = float(target_value)

    if not isinstance(tolerance, (int, float)) or float(tolerance) < 0:
        raise ValueError("tolerance must be a non-negative number")
    tol = float(tolerance)

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    # Get a reasonably large number of initial matches to filter from
    base_matches = get_molecules_by_name(query=query, limit=max(limit * 5, limit))

    results: List[Dict[str, Any]] = []
    for rec in base_matches:
        props = rec.get("properties", {})
        if key not in props:
            continue
        value = float(props[key])
        diff = abs(value - target)
        if tol == 0 and diff != 0:
            continue
        if tol > 0 and diff > tol:
            continue
        score = _score_match(value, target, tol)
        justification = (
            f"Selected because {key} is {value:.3f}, which is within ±{tol} of the target {target:.3f} (difference {diff:.3f})."
        )
        results.append({
            "name": rec.get("name"),
            "smiles": rec.get("smiles"),
            "properties": props,
            "property_name": key,
            "target_value": target,
            "value": value,
            "difference": diff,
            "match_quality": score,
            "justification": justification,
        })

    results.sort(key=lambda r: (r["difference"], -r["match_quality"]))
    return results[:limit]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Molecular Property Analysis Agent (RDKit)")
    p.add_argument("--query", required=True, help="Molecule name or substring to search for (case-insensitive)")
    p.add_argument("--property", required=True, dest="property_name", choices=list(SUPPORTED_PROPERTIES.keys()), help="Property to target")
    p.add_argument("--target", required=True, type=float, dest="target_value", help="Target numeric value for the property")
    p.add_argument("--tolerance", type=float, default=0.5, help="Allowed absolute deviation from target (default: 0.5)")
    p.add_argument("--limit", type=int, default=10, help="Maximum number of results to return (default: 10)")
    p.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to dataset JSON (default: /app/resources/test_molecules.json)")
    p.add_argument("--as-json", action="store_true", help="Output results as JSON (default)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Optionally override dataset path if provided
    global DEFAULT_DATA_PATH
    if args.data:
        DEFAULT_DATA_PATH = args.data

    try:
        results = find_candidates_for_property(
            property_name=args.property_name,
            target_value=args.target_value,
            query=args.query,
            tolerance=args.tolerance,
            limit=args.limit,
        )
        output = {
            "query": args.query,
            "property": args.property_name,
            "target": args.target_value,
            "tolerance": args.tolerance,
            "limit": args.limit,
            "count": len(results),
            "results": results,
        }
        print(json.dumps(output, indent=2))
        return 0
    except FileNotFoundError as e:
        err = {"error": "dataset_not_found", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 1
    except ValueError as e:
        err = {"error": "invalid_input", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 2
    except ChemAgentError as e:
        err = {"error": "chem_agent_error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 3
    except Exception as e:  # pragma: no cover - unexpected errors
        err = {"error": "unexpected_error", "message": str(e)}
        print(json.dumps(err, indent=2), file=sys.stderr)
        return 99


if __name__ == "__main__":
    raise SystemExit(main())
