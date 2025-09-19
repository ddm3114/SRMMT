#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

DEFAULT_DATA_PATH = "/app/resources/test_molecules.json"


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coalesce_smiles(record: Dict[str, Any]) -> Optional[str]:
    # Try common key variants for SMILES
    for key in ("smiles", "SMILES", "canonical_smiles", "CanonicalSMILES", "canonicalSmiles"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key].strip()
    return None


def smiles_to_mol(smiles: str) -> Optional[object]:
    """
    Converts a SMILES string to an RDKit molecule object.
    Returns None if the SMILES is invalid or RDKit is not available.
    """
    try:
        from rdkit import Chem  # type: ignore
    except Exception:
        return None

    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception:
        return None


def compute_properties(mol: object) -> Dict[str, Any]:
    """
    Compute key molecular properties using RDKit for the provided molecule.
    Returns a dictionary with: logp, mw, donors, acceptors, rotatable_bonds, tpsa
    """
    if mol is None:
        raise ValueError("compute_properties requires a valid RDKit Mol object")
    try:
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # type: ignore
    except Exception as e:
        raise RuntimeError("RDKit is required to compute properties. Please ensure rdkit is installed.") from e

    try:
        props: Dict[str, Any] = {
            "logp": float(Crippen.MolLogP(mol)),
            "mw": float(Descriptors.MolWt(mol)),
            "donors": int(Lipinski.NumHDonors(mol)),
            "acceptors": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        }
    except Exception as e:
        raise RuntimeError("Failed to compute properties for the provided molecule.") from e

    return props


def _normalize_records(data: Any) -> List[Dict[str, Any]]:
    # Accept either a list of molecules or a dict containing such a list
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # common container keys
        for key in ("molecules", "Molecules", "data", "items", "records"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                break
        else:
            # Treat the dict itself as a single record if it resembles a molecule
            records = [data]
    else:
        records = []

    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        name = rec.get("name") or rec.get("Name") or rec.get("title") or rec.get("Title")
        if isinstance(name, str):
            name = name.strip()
        smiles = _coalesce_smiles(rec)
        props = rec.get("properties") if isinstance(rec.get("properties"), dict) else None

        normalized.append({
            "name": name,
            "smiles": smiles,
            "properties": props,
            **{k: v for k, v in rec.items() if k not in {"name", "Name", "title", "Title", "smiles", "SMILES", "canonical_smiles", "CanonicalSMILES", "canonicalSmiles", "properties"}},
        })
    return normalized


def _load_dataset(explicit_path: Optional[str] = None) -> List[Dict[str, Any]]:
    # Try explicit path, then default, then a few common fallbacks
    candidate_paths = [p for p in [explicit_path, DEFAULT_DATA_PATH] if p]

    # Relative fallbacks within repo
    here = os.path.dirname(os.path.abspath(__file__))
    candidate_paths.extend([
        os.path.join(here, "resources", "test_molecules.json"),
        os.path.join(here, "data", "test_molecules.json"),
        os.path.join("/app", "resources", "test_molecules.json"),
    ])

    last_err = None
    for path in candidate_paths:
        try:
            data = _read_json(path)
            return _normalize_records(data)
        except FileNotFoundError as e:
            last_err = e
            continue
        except json.JSONDecodeError as e:
            # If path exists but content invalid, propagate
            raise
        except Exception as e:
            last_err = e
            continue

    msg = f"Could not load molecule dataset. Tried paths: {candidate_paths}"
    if last_err:
        msg += f". Last error: {last_err}"
    raise FileNotFoundError(msg)


def get_molecules_by_name(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Loads molecules from the local test data file and filters by name using case-insensitive matching.
    Returns up to `limit` molecules with fields: name, smiles, properties.
    If properties are missing in the dataset, they are computed via RDKit when possible.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    records = _load_dataset()

    q = query.lower()
    matches: List[Dict[str, Any]] = []

    for rec in records:
        name = rec.get("name")
        if not isinstance(name, str):
            continue
        if q in name.lower():
            smiles = rec.get("smiles")
            props = rec.get("properties") if isinstance(rec.get("properties"), dict) else None

            if props is None and isinstance(smiles, str):
                mol = smiles_to_mol(smiles)
                if mol is not None:
                    try:
                        props = compute_properties(mol)
                    except Exception:
                        props = None

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
    valid = {
        "logp": "logp",
        "mw": "mw",
        "molecular_weight": "mw",
        "donors": "donors",
        "hbd": "donors",
        "acceptors": "acceptors",
        "hba": "acceptors",
        "rotatable_bonds": "rotatable_bonds",
        "rotb": "rotatable_bonds",
        "tpsa": "tpsa",
    }
    if key not in valid:
        raise ValueError(f"Unsupported property_name: {property_name}. Supported: {sorted(set(valid.values()))}")
    return valid[key]


def find_candidates_for_property(
    property_name: str,
    target_value: float,
    query: str,
    tolerance: float = 0.5,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Retrieve molecules matching the query and filter those that have the specified property
    within the tolerance range around the target value. Results are sorted by proximity and
    annotated with match_quality (1.0 at perfect match, 0.0 at tolerance boundary).
    """
    prop_key = _validate_property_name(property_name)

    if not isinstance(target_value, (int, float)) or not math.isfinite(float(target_value)):
        raise ValueError("target_value must be a finite number")
    target_value = float(target_value)

    if not isinstance(tolerance, (int, float)) or tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be a non-negative finite number")
    tolerance = float(tolerance)

    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")

    # Fetch a generous number to avoid missing candidates before filtering
    base_matches = get_molecules_by_name(query, limit=max(limit * 10, 50))

    results: List[Dict[str, Any]] = []

    for item in base_matches:
        props = item.get("properties") or {}
        if not isinstance(props, dict) or prop_key not in props:
            # Try to compute if smiles present
            smiles = item.get("smiles")
            if isinstance(smiles, str):
                mol = smiles_to_mol(smiles)
                if mol is not None:
                    try:
                        props = compute_properties(mol)
                    except Exception:
                        props = {}
        if prop_key not in props:
            continue
        try:
            val = float(props[prop_key])
        except Exception:
            continue
        diff = abs(val - target_value)
        if diff <= tolerance:
            quality = 1.0 if tolerance == 0 else max(0.0, 1.0 - (diff / tolerance))
            results.append({
                "name": item.get("name"),
                "smiles": item.get("smiles"),
                "properties": props,
                "target_property": prop_key,
                "target_value": target_value,
                "actual_value": val,
                "difference": diff,
                "within_tolerance": True,
                "match_quality": quality,
                "justification": (
                    f"{item.get('name')} selected because {prop_key}={val:.3f} "
                    f"is within ±{tolerance} of target {target_value}."
                ),
            })

    results.sort(key=lambda r: (r.get("difference", float("inf")), -r.get("match_quality", 0.0)))

    return results[:limit]


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Molecular property analysis agent that searches molecules and filters by properties.",
    )
    parser.add_argument("--query", required=True, help="Query name or substring to search for (e.g., 'aspirin')")
    parser.add_argument("--property", dest="property_name", required=True, help="Property to filter by: logp, mw, donors, acceptors, rotatable_bonds, tpsa")
    parser.add_argument("--target", dest="target_value", type=float, required=True, help="Target value for the chosen property")
    parser.add_argument("--tolerance", type=float, default=0.5, help="Tolerance range around the target value (default: 0.5)")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of results to return (default: 10)")
    parser.add_argument("--data", dest="data_path", default=None, help="Optional path to a JSON file containing molecules")

    args = parser.parse_args()

    # If a custom data path is provided, we temporarily override the loader by patching environment
    results: List[Dict[str, Any]]
    try:
        if args.data_path:
            # Monkey-patch loader to prioritize explicit path
            def _load_with_explicit():
                return _load_dataset(args.data_path)
            # Gather a broader initial set by temporarily intercepting get_molecules_by_name
            def get_molecules_by_name_with_data(query: str, limit: int = 5) -> List[Dict[str, Any]]:
                if not isinstance(query, str) or not query.strip():
                    raise ValueError("query must be a non-empty string")
                if not isinstance(limit, int) or limit <= 0:
                    raise ValueError("limit must be a positive integer")
                records = _load_with_explicit()
                q = query.lower()
                matches: List[Dict[str, Any]] = []
                for rec in records:
                    name = rec.get("name")
                    if not isinstance(name, str):
                        continue
                    if q in name.lower():
                        smiles = rec.get("smiles")
                        props = rec.get("properties") if isinstance(rec.get("properties"), dict) else None
                        if props is None and isinstance(smiles, str):
                            mol = smiles_to_mol(smiles)
                            if mol is not None:
                                try:
                                    props = compute_properties(mol)
                                except Exception:
                                    props = None
                        matches.append({"name": name, "smiles": smiles, "properties": props})
                        if len(matches) >= limit:
                            break
                return matches
            # Use our local version for this CLI invocation only
            base_matches = get_molecules_by_name_with_data(args.query, limit=max(args.limit * 10, 50))
            # Filter
            prop_key = _validate_property_name(args.property_name)
            if not math.isfinite(args.tolerance) or args.tolerance < 0:
                raise ValueError("tolerance must be a non-negative finite number")
            results = []
            for item in base_matches:
                props = item.get("properties") or {}
                if not isinstance(props, dict) or prop_key not in props:
                    smiles = item.get("smiles")
                    if isinstance(smiles, str):
                        mol = smiles_to_mol(smiles)
                        if mol is not None:
                            try:
                                props = compute_properties(mol)
                            except Exception:
                                props = {}
                if prop_key not in props:
                    continue
                try:
                    val = float(props[prop_key])
                except Exception:
                    continue
                diff = abs(val - float(args.target_value))
                if diff <= float(args.tolerance):
                    quality = 1.0 if float(args.tolerance) == 0 else max(0.0, 1.0 - (diff / float(args.tolerance)))
                    results.append({
                        "name": item.get("name"),
                        "smiles": item.get("smiles"),
                        "properties": props,
                        "target_property": prop_key,
                        "target_value": float(args.target_value),
                        "actual_value": val,
                        "difference": diff,
                        "within_tolerance": True,
                        "match_quality": quality,
                        "justification": (
                            f"{item.get('name')} selected because {prop_key}={val:.3f} "
                            f"is within ±{args.tolerance} of target {args.target_value}."
                        ),
                    })
            results.sort(key=lambda r: (r.get("difference", float("inf")), -r.get("match_quality", 0.0)))
            results = results[: int(args.limit)]
        else:
            results = find_candidates_for_property(
                property_name=args.property_name,
                target_value=float(args.target_value),
                query=args.query,
                tolerance=float(args.tolerance),
                limit=int(args.limit),
            )

        output = {"results": results, "count": len(results)}
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    except Exception as e:
        err = {
            "error": str(e),
            "type": type(e).__name__,
        }
        json.dump(err, sys.stderr)
        sys.stderr.write("\n")
        return 1


if __name__ == "__main__":
    sys.exit(_cli())
