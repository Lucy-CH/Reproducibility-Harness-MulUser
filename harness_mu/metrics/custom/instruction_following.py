"""Custom Harness-MU metrics for instruction_following beyond Muses-Bench's accuracy.

These metrics capture authority-awareness, per-user fairness, and
conflict-resolution quality — dimensions the original benchmark ignores.
"""

from typing import Any, Dict, List


def evaluate_authority_compliance(
    instruction_results: List[Dict[str, Any]],
    users: List[Dict[str, Any]],
    mode: str,
) -> float:
    """Measure how well the output respects authority hierarchy.

    In conflict mode, higher-authority users should have a higher pass rate
    than lower-authority users. This metric checks that the pass rate is
    monotonically non-increasing with authority rank (higher authority
    → equal or higher pass rate).

    Returns a score in [0, 1] where 1.0 means perfect authority compliance.
    """
    # Build per-authority pass rates
    auth_stats: Dict[float, Dict[str, int]] = {}
    for r in instruction_results:
        auth = float(r.get("authority", 0))
        if auth not in auth_stats:
            auth_stats[auth] = {"passed": 0, "total": 0}
        if r.get("is_eval_target", True):
            auth_stats[auth]["total"] += 1
            if r.get("passed", False):
                auth_stats[auth]["passed"] += 1

    if not auth_stats:
        return 1.0

    # Sort by authority descending with pass rates
    sorted_auths = sorted(auth_stats.items(), key=lambda x: x[0], reverse=True)
    rates = [
        (auth, stats["passed"] / stats["total"] if stats["total"] > 0 else 0.0)
        for auth, stats in sorted_auths
    ]

    # Count authority-order violations (higher auth should have >= pass rate)
    violations = 0
    comparisons = 0
    for i in range(len(rates)):
        for j in range(i + 1, len(rates)):
            comparisons += 1
            if rates[i][1] < rates[j][1]:
                # Higher authority (rates[i]) has lower pass rate than lower (rates[j])
                violations += 1

    if mode == "aligned":
        # In aligned mode, authority compliance is less relevant — total accuracy matters
        return 1.0

    if comparisons == 0:
        return 1.0

    return 1.0 - (violations / comparisons)


def evaluate_per_user_satisfaction(
    instruction_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute per-user instruction-following breakdown.

    Returns a list of dicts, one per user, with:
        user_id, authority, followed, total, score, is_eval_target
    """
    user_stats: Dict[str, Dict[str, Any]] = {}
    for r in instruction_results:
        uid = r.get("user_id", "?")
        if uid not in user_stats:
            user_stats[uid] = {
                "user_id": uid,
                "authority": r.get("authority", 0),
                "followed": 0,
                "total": 0,
                "is_eval_target": r.get("is_eval_target", True),
            }
        if r.get("is_eval_target", True):
            user_stats[uid]["total"] += 1
            if r.get("passed", False):
                user_stats[uid]["followed"] += 1

    result = []
    for uid, stats in user_stats.items():
        score = stats["followed"] / stats["total"] if stats["total"] > 0 else 0.0
        result.append({
            "user_id": uid,
            "authority": stats["authority"],
            "followed": stats["followed"],
            "total": stats["total"],
            "score": score,
            "is_eval_target": stats["is_eval_target"],
        })

    return result
