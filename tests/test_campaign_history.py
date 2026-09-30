from cocofest.simulation.campaign_history import CampaignHistory


def test_campaign_history_persists_launch_details_and_terminal_status(tmp_path):
    history = CampaignHistory(tmp_path / "gui-results" / "campaign-history.json")
    entry_id = history.add(kind="Deux bras indépendants", solver="ipopt", formulation="isokinetic",
                           cycles="100", controls="30", cpu="D:30 G:31", output_root="results/run")
    assert history.entries()[0]["status"] == "en cours"
    history.set_status(entry_id, "terminée")
    entry = CampaignHistory(history.path).entries()[0]
    assert entry["cpu"] == "D:30 G:31"
    assert entry["status"] == "terminée"
    assert entry["ended_at"]
