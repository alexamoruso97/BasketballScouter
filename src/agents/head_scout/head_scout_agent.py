"""Head Scout Agent: Tactical action attribution, boxscore calculation, and ground truth validation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


class HeadScoutAgent:
    """Analyzes possession clips, attributes points and assists, and compiles the final boxscore."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.boxscore: Dict[str, Dict[str, Dict[str, Any]]] = {
            "HOME": {},
            "AWAY": {}
        }
        self.attributions: List[Dict[str, Any]] = []

    def _ensure_player_entry(self, team: str, player_name: str, jersey_num: Optional[int]) -> Dict[str, Any]:
        """Initializes or retrieves a player's boxscore entry."""
        key = f"#{jersey_num} {player_name}" if jersey_num is not None else player_name
        if key not in self.boxscore[team]:
            self.boxscore[team][key] = {
                "jersey": jersey_num,
                "name": player_name,
                "team": team,
                "pts": 0,
                "2pm": 0,
                "3pm": 0,
                "ftm": 0,
                "ast": 0,
                "possessions_involved": 0
            }
        return self.boxscore[team][key]

    def attribute_possession(
        self,
        clip_profile: Dict[str, Any],
        score_event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attributes a scored possession to the shooter and potential assist provider."""
        clip_id = clip_profile.get("clip_id", "UNKNOWN")
        clip_file = clip_profile.get("clip_file", "")
        scoring_team = score_event.get("team", "HOME")
        points_delta = int(score_event.get("points_delta", 2))
        timestamp_sec = score_event.get("timestamp_sec", 0.0)

        # Candidates belonging to the scoring team
        team_players = (
            clip_profile.get("home_players", [])
            if scoring_team == "HOME"
            else clip_profile.get("away_players", [])
        )

        # Identified candidate players with confirmed jersey numbers
        identified = [p for p in team_players if p.get("jersey") is not None]

        shooter_name = "Team Unassigned"
        shooter_jersey = None
        assist_name = None
        assist_jersey = None

        if identified:
            # Primary shooter: first confirmed offensive player prominent in action
            primary = identified[0]
            shooter_name = primary.get("name", "Unknown")
            shooter_jersey = primary.get("jersey")

            # Update shooter boxscore
            entry = self._ensure_player_entry(scoring_team, shooter_name, shooter_jersey)
            entry["pts"] += points_delta
            entry["possessions_involved"] += 1
            if points_delta == 1:
                entry["ftm"] += 1
            elif points_delta == 2:
                entry["2pm"] += 1
            elif points_delta == 3:
                entry["3pm"] += 1

            # Assist provider: secondary teammate on made field goals (2PT or 3PT)
            if len(identified) >= 2 and points_delta in [2, 3]:
                secondary = identified[1]
                assist_name = secondary.get("name", "Unknown")
                assist_jersey = secondary.get("jersey")
                ast_entry = self._ensure_player_entry(scoring_team, assist_name, assist_jersey)
                ast_entry["ast"] += 1
                ast_entry["possessions_involved"] += 1
        else:
            # Unidentified track: record team points
            entry = self._ensure_player_entry(scoring_team, f"{scoring_team} Offensive Player", None)
            entry["pts"] += points_delta
            entry["possessions_involved"] += 1
            if points_delta == 1:
                entry["ftm"] += 1
            elif points_delta == 2:
                entry["2pm"] += 1
            elif points_delta == 3:
                entry["3pm"] += 1

        attribution = {
            "clip_id": clip_id,
            "clip_file": clip_file,
            "timestamp_sec": timestamp_sec,
            "scoring_team": scoring_team,
            "points": points_delta,
            "shooter": {"jersey": shooter_jersey, "name": shooter_name},
            "assist": {"jersey": assist_jersey, "name": assist_name} if assist_name else None
        }

        self.attributions.append(attribution)
        return attribution

    def attribute_all_possessions(
        self,
        profiles_list: List[Dict[str, Any]],
        manifest_path: str | Path
    ) -> List[Dict[str, Any]]:
        """Iterates over all profiled clips and matches them with their manifest scoring events."""
        manifest_path = Path(manifest_path)
        manifest_map: Dict[str, Dict[str, Any]] = {}

        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_items = json.load(f)
            for item in manifest_items:
                manifest_map[item.get("clip_id", "")] = item.get("score_event", {})

        for profile in profiles_list:
            cid = profile.get("clip_id", "")
            score_event = manifest_map.get(cid)
            if not score_event:
                # Default fallback
                score_event = {"team": "HOME", "points_delta": 2, "timestamp_sec": 0.0}
            self.attribute_possession(profile, score_event)

        return self.attributions

    def evaluate_against_ground_truth(self, ground_truth_json_path: str | Path) -> Dict[str, Any]:
        """Compares calculated boxscore with official LBA Ground Truth dataset."""
        gt_path = Path(ground_truth_json_path)
        if not gt_path.exists():
            return {"status": "NO_GROUND_TRUTH_FILE"}

        with open(gt_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

        gt_scores = gt_data.get("game", {}).get("scores", {})
        comparisons = []

        for team_key, gt_team_key in [("HOME", "ht"), ("AWAY", "vt")]:
            gt_rows = gt_scores.get(gt_team_key, {}).get("rows", [])
            for p in gt_rows:
                num = p.get("player_num")
                surn = p.get("player_surname", "")
                official_pts = p.get("pun", 0)
                official_ast = p.get("ass", 0)

                # Match in our computed boxscore
                matched_scouter = None
                for player_key, sc_p in self.boxscore[team_key].items():
                    if sc_p.get("jersey") == num or surn.upper() in sc_p.get("name", "").upper():
                        matched_scouter = sc_p
                        break

                detected_pts = matched_scouter.get("pts", 0) if matched_scouter else 0
                detected_ast = matched_scouter.get("ast", 0) if matched_scouter else 0

                comparisons.append({
                    "team": team_key,
                    "jersey": num,
                    "player": f"{surn} {p.get('player_name', '')}",
                    "official_pts": official_pts,
                    "detected_pts": detected_pts,
                    "pts_diff": detected_pts - official_pts,
                    "official_ast": official_ast,
                    "detected_ast": detected_ast
                })

        return {"comparisons": comparisons}

    def generate_final_report(self, ground_truth_json_path: Optional[str | Path] = None) -> Path:
        """Saves final_boxscore.json and compiles comprehensive scouting_report.md."""
        boxscore_file = self.output_dir / "final_boxscore.json"
        with open(boxscore_file, "w", encoding="utf-8") as f:
            json.dump({
                "boxscore": self.boxscore,
                "attributions": self.attributions
            }, f, indent=2)

        report_file = self.output_dir / "scouting_report.md"

        lines = [
            "# Scouter Automated Match Scouting Report",
            "",
            "## 1. Boxscore: Virtus Segafredo Bologna (HOME)",
            "| Player | PTS | 2PM | 3PM | FTM | AST | Possessions Involved |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        for p in self.boxscore["HOME"].values():
            p_label = f"#{p['jersey']} {p['name']}" if p.get("jersey") is not None else p['name']
            lines.append(f"| **{p_label}** | **{p['pts']}** | {p['2pm']} | {p['3pm']} | {p['ftm']} | {p['ast']} | {p['possessions_involved']} |")

        lines.extend([
            "",
            "## 2. Boxscore: EA7 Emporio Armani Milano (AWAY)",
            "| Player | PTS | 2PM | 3PM | FTM | AST | Possessions Involved |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for p in self.boxscore["AWAY"].values():
            p_label = f"#{p['jersey']} {p['name']}" if p.get("jersey") is not None else p['name']
            lines.append(f"| **{p_label}** | **{p['pts']}** | {p['2pm']} | {p['3pm']} | {p['ftm']} | {p['ast']} | {p['possessions_involved']} |")

        lines.extend([
            "",
            "## 3. Scored Possession Attributions (Sample Actions)",
            "| Clip ID | Target Time | Team | Points | Shooter Credited | Assist Credited |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for att in self.attributions:
            sh = att.get("shooter", {})
            sh_label = f"#{sh.get('jersey')} {sh.get('name')}" if sh.get("jersey") is not None else sh.get("name", "Unknown")
            ast = att.get("assist")
            ast_label = f"#{ast.get('jersey')} {ast.get('name')}" if (ast and ast.get("jersey") is not None) else (ast.get("name") if ast else "-")
            m = int(att.get("timestamp_sec", 0.0) // 60)
            s = int(att.get("timestamp_sec", 0.0) % 60)
            lines.append(f"| `{att.get('clip_id')}` | {m:02d}:{s:02d} | **{att.get('scoring_team')}** | +{att.get('points')} | **{sh_label}** | {ast_label} |")

        # Ground Truth Validation Comparison if available
        if ground_truth_json_path:
            eval_res = self.evaluate_against_ground_truth(ground_truth_json_path)
            comparisons = eval_res.get("comparisons", [])
            if comparisons:
                lines.extend([
                    "",
                    "## 4. Ground Truth Validation Audit (Scouter vs Official LBA Data)",
                    "| Team | Jersey | Player | Scouter PTS | Official PTS | Diff | Scouter AST | Official AST |",
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
                ])
                for cmp in comparisons:
                    lines.append(
                        f"| {cmp['team']} | #{cmp['jersey']} | {cmp['player']} | **{cmp['detected_pts']}** | {cmp['official_pts']} | {cmp['pts_diff']:+d} | {cmp['detected_ast']} | {cmp['official_ast']} |"
                    )

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_file
