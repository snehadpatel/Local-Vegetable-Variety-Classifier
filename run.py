import http.server
import socketserver
import json
import os
import shutil
import webbrowser
import urllib.parse
import threading
import time

PORT = 5000
DATASET_DIR = "dataset/peas"
REMOVED_DIR = "dataset/peas_removed"
STATE_FILE = "dataset_cleaner_state.json"

# State variables
master_list = []
kept_set = set()
history = []
current_index = 0

def init_state():
    global master_list, kept_set, history, current_index
    
    # Create directories if they don't exist
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    if not os.path.exists(REMOVED_DIR):
        os.makedirs(REMOVED_DIR)
        
    # Build master list of all original images (peas + peas_removed)
    active_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    removed_files = [f for f in os.listdir(REMOVED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
    
    # Filter out hidden files
    active_files = [f for f in active_files if not f.startswith(".")]
    removed_files = [f for f in removed_files if not f.startswith(".")]
    
    master_list = sorted(list(set(active_files + removed_files)))
    
    # Load state from file if exists
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
                current_index = saved.get("currentIndex", 0)
                kept_set = set(saved.get("kept", []))
                history = saved.get("history", [])
        except Exception as e:
            print("Error loading state file:", e)
            current_index = 0
            kept_set = set()
            history = []
    else:
        current_index = 0
        kept_set = set()
        history = []
        
    # Alignment: Make sure file location on disk matches state
    removed_in_history = {item["filename"] for item in history if item["action"] == "remove"}
    for filename in master_list:
        in_active = os.path.exists(os.path.join(DATASET_DIR, filename))
        in_removed = os.path.exists(os.path.join(REMOVED_DIR, filename))
        
        should_be_removed = filename in removed_in_history
        
        if should_be_removed and in_active:
            shutil.move(os.path.join(DATASET_DIR, filename), os.path.join(REMOVED_DIR, filename))
        elif not should_be_removed and in_removed:
            shutil.move(os.path.join(REMOVED_DIR, filename), os.path.join(DATASET_DIR, filename))

    # Guard current_index out of bounds
    if current_index < 0:
        current_index = 0
    if len(master_list) > 0 and current_index > len(master_list):
        current_index = len(master_list)

def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({
                "currentIndex": current_index,
                "kept": list(kept_set),
                "history": history
            }, f, indent=2)
    except Exception as e:
        print("Error saving state:", e)

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dataset Cleaner - Peas</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0B0F19;
            --bg-card: rgba(22, 30, 49, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --primary: #8B5CF6;
            --keep: #10B981;
            --keep-hover: #059669;
            --remove: #F43F5E;
            --remove-hover: #E11D48;
            --undo: #F59E0B;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
            -webkit-user-select: none;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: 
                radial-gradient(at 10% 20%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(244, 63, 94, 0.1) 0px, transparent 50%);
        }

        header {
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border-color);
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(12px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }

        .logo-section {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-indicator {
            width: 10px;
            height: 10px;
            background-color: var(--keep);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--keep);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 16px var(--keep); }
            100% { transform: scale(0.95); opacity: 0.5; }
        }

        h1 {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #FFF 30%, var(--primary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .dataset-badge {
            background: rgba(139, 92, 246, 0.2);
            border: 1px solid rgba(139, 92, 246, 0.3);
            color: #C084FC;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }

        .main-container {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 380px;
            gap: 2rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
            width: 100%;
        }

        /* Image Display Section */
        .workspace {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            min-width: 0; /* Prevents overflow */
        }

        .image-card {
            flex: 1;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            position: relative;
            backdrop-filter: blur(20px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .image-card.flash-keep {
            border-color: var(--keep);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.3);
        }

        .image-card.flash-remove {
            border-color: var(--remove);
            box-shadow: 0 0 30px rgba(244, 63, 148, 0.3);
        }

        .image-header {
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid var(--border-color);
            font-size: 0.85rem;
            font-weight: 500;
        }

        .image-name {
            color: var(--text-main);
            font-family: monospace;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 60%;
        }

        .image-index {
            color: var(--text-muted);
        }

        .image-viewport {
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            position: relative;
            background: rgba(0, 0, 0, 0.15);
            min-height: 450px;
        }

        .active-image {
            max-width: 100%;
            max-height: 520px;
            object-fit: contain;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
            transition: transform 0.2s ease, opacity 0.2s ease;
        }

        /* Sidebar Section */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .panel {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1.5rem;
            backdrop-filter: blur(20px);
        }

        .panel-title {
            font-size: 0.9rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Stats grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }

        .stat-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
        }

        .stat-value {
            font-size: 1.75rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.25rem;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        .stat-card.keep-stat .stat-value { color: var(--keep); }
        .stat-card.remove-stat .stat-value { color: var(--remove); }

        /* Controls Panel */
        .controls-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .btn {
            border: none;
            border-radius: 16px;
            padding: 1rem;
            font-family: inherit;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            color: white;
            position: relative;
            overflow: hidden;
        }

        .btn-keep {
            background: var(--keep);
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2);
        }

        .btn-keep:hover {
            background: var(--keep-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3);
        }

        .btn-remove {
            background: var(--remove);
            box-shadow: 0 4px 12px rgba(244, 63, 94, 0.2);
        }

        .btn-remove:hover {
            background: var(--remove-hover);
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(244, 63, 94, 0.3);
        }

        .btn-undo {
            grid-column: span 2;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: var(--undo);
            padding: 0.75rem;
        }

        .btn-undo:hover:not(:disabled) {
            background: rgba(245, 158, 11, 0.25);
            transform: translateY(-1px);
        }

        .btn-undo:disabled {
            opacity: 0.3;
            cursor: not-allowed;
        }

        .shortcut-key {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 0.1rem 0.5rem;
            border-radius: 6px;
            font-size: 0.7rem;
            font-family: monospace;
            color: rgba(255,255,255,0.7);
        }

        /* Keyboard Panel */
        .shortcut-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(255,255,255,0.04);
            font-size: 0.85rem;
        }
        
        .shortcut-row:last-child {
            border-bottom: none;
        }

        .shortcut-desc {
            color: var(--text-muted);
        }

        .shortcut-keys {
            display: flex;
            gap: 0.25rem;
        }

        /* History panel */
        .history-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: 240px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }

        .history-list::-webkit-scrollbar {
            width: 4px;
        }

        .history-list::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.1);
            border-radius: 2px;
        }

        .history-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0.8rem;
            background: rgba(0, 0, 0, 0.15);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            font-size: 0.8rem;
        }

        .history-info {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: monospace;
            max-width: 75%;
            overflow: hidden;
        }

        .history-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .history-dot.keep { background-color: var(--keep); }
        .history-dot.remove { background-color: var(--remove); }

        .history-name {
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .history-action {
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
        }

        .history-action.keep {
            background: rgba(16, 185, 129, 0.15);
            color: var(--keep);
        }

        .history-action.remove {
            background: rgba(244, 63, 94, 0.15);
            color: var(--remove);
        }

        /* Progress Bar */
        .progress-section {
            padding: 0 2rem;
            max-width: 1400px;
            margin: 0 auto 1.5rem auto;
            width: 100%;
        }

        .progress-bar-container {
            width: 100%;
            height: 8px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 9999px;
            overflow: hidden;
            position: relative;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, var(--primary) 0%, var(--keep) 100%);
            border-radius: 9999px;
            width: 0%;
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
        }

        .progress-bar::after {
            content: '';
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            width: 10px;
            background: white;
            filter: blur(4px);
            opacity: 0.5;
        }

        .progress-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .progress-percent {
            color: var(--primary);
            font-weight: 700;
        }

        /* Success screen */
        .completion-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 4rem 2rem;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            max-width: 600px;
            margin: 4rem auto;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            backdrop-filter: blur(20px);
        }

        .completion-icon {
            font-size: 4rem;
            margin-bottom: 1.5rem;
            animation: bounce 2s infinite;
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }

        .completion-title {
            font-size: 1.75rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, #FFF 30%, var(--keep) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .completion-desc {
            color: var(--text-muted);
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        .btn-restart {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-color);
            color: var(--text-main);
            padding: 0.75rem 2rem;
            font-size: 0.9rem;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-restart:hover {
            background: rgba(255,255,255,0.15);
            transform: translateY(-1px);
        }

        /* Empty list warning */
        .empty-warning {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            height: 100%;
            color: var(--text-muted);
            gap: 1rem;
        }

        .empty-warning svg {
            width: 48px;
            height: 48px;
            opacity: 0.5;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo-section">
            <div class="logo-indicator"></div>
            <h1>Vegetable Classifier</h1>
            <span class="dataset-badge">Peas Dataset</span>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-muted); font-weight: 500;">
            Local Dataset Cleaner
        </div>
    </header>

    <div id="active-app-view" style="display: none; flex-direction: column; flex: 1;">
        <div class="main-container">
            <!-- Workspace: Image preview -->
            <div class="workspace">
                <div class="image-card" id="main-image-card">
                    <div class="image-header">
                        <span class="image-name" id="display-filename">loading...</span>
                        <span class="image-index" id="display-index">Image 0 of 0</span>
                    </div>
                    <div class="image-viewport">
                        <img src="" id="display-img" class="active-image" alt="Classified vegetable specimen">
                    </div>
                </div>
            </div>

            <!-- Sidebar: Stats and controls -->
            <div class="sidebar">
                <!-- Stats panel -->
                <div class="panel">
                    <div class="panel-title">Overview</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="stat-total">0</div>
                            <div class="stat-label">Total Images</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" style="color: var(--primary);" id="stat-processed">0</div>
                            <div class="stat-label">Processed</div>
                        </div>
                        <div class="stat-card keep-stat">
                            <div class="stat-value" id="stat-keep">0</div>
                            <div class="stat-label">Kept</div>
                        </div>
                        <div class="stat-card remove-stat">
                            <div class="stat-value" id="stat-remove">0</div>
                            <div class="stat-label">Removed</div>
                        </div>
                    </div>
                </div>

                <!-- Controls panel -->
                <div class="panel">
                    <div class="panel-title">Actions</div>
                    <div class="controls-grid">
                        <button class="btn btn-keep" onclick="triggerAction('keep')">
                            <span>KEEP</span>
                            <span class="shortcut-key">K</span>
                        </button>
                        <button class="btn btn-remove" onclick="triggerAction('remove')">
                            <span>REMOVE</span>
                            <span class="shortcut-key">R</span>
                        </button>
                        <button class="btn btn-undo" id="btn-undo-control" onclick="triggerAction('undo')" disabled>
                            <span>UNDO LAST ACTION</span>
                            <span class="shortcut-key" style="margin-left: 0.5rem;">Z</span>
                        </button>
                    </div>
                    
                    <div style="margin-top: 1rem; border-top: 1px solid var(--border-color); padding-top: 1rem;">
                        <div class="shortcut-row">
                            <span class="shortcut-desc">Keep Image</span>
                            <div class="shortcut-keys">
                                <span class="shortcut-key">K</span>
                                <span class="shortcut-key">→</span>
                            </div>
                        </div>
                        <div class="shortcut-row">
                            <span class="shortcut-desc">Remove Image</span>
                            <div class="shortcut-keys">
                                <span class="shortcut-key">R</span>
                                <span class="shortcut-key">←</span>
                                <span class="shortcut-key">Del</span>
                            </div>
                        </div>
                        <div class="shortcut-row">
                            <span class="shortcut-desc">Undo Action</span>
                            <div class="shortcut-keys">
                                <span class="shortcut-key">Z</span>
                                <span class="shortcut-key">Backspace</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- History panel -->
                <div class="panel" style="flex: 1; display: flex; flex-direction: column;">
                    <div class="panel-title">
                        <span>Recent Actions</span>
                        <span style="font-size: 0.75rem; text-transform: none; font-weight: normal;" id="history-count">(0)</span>
                    </div>
                    <div class="history-list" id="history-container">
                        <!-- History items injected here -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom progress section -->
        <div class="progress-section">
            <div class="progress-label">
                <span style="color: var(--text-muted);">Cleaning Progress</span>
                <span class="progress-percent" id="progress-percent-label">0%</span>
            </div>
            <div class="progress-bar-container">
                <div class="progress-bar" id="app-progress-bar"></div>
            </div>
        </div>
    </div>

    <!-- Completion view -->
    <div id="completion-view" style="display: none; flex: 1; justify-content: center; align-items: center;">
        <div class="completion-container">
            <div class="completion-icon">🎉</div>
            <h2 class="completion-title">Dataset Cleaned!</h2>
            <p class="completion-desc">
                Congratulations! You have completed reviewing all images in the dataset folder.
                Your cleaned dataset is perfectly prepared in <span style="font-family: monospace; color: white;">dataset/peas/</span>.
                Removed files have been safely isolated in <span style="font-family: monospace; color: white;">dataset/peas_removed/</span>.
            </p>
            <div class="stats-grid" style="width: 100%; margin-bottom: 2rem; max-width: 400px; margin-left: auto; margin-right: auto;">
                <div class="stat-card">
                    <div class="stat-value" id="complete-total">0</div>
                    <div class="stat-label">Reviewed</div>
                </div>
                <div class="stat-card keep-stat">
                    <div class="stat-value" id="complete-keep">0</div>
                    <div class="stat-label">Kept</div>
                </div>
                <div class="stat-card remove-stat">
                    <div class="stat-value" id="complete-remove">0</div>
                    <div class="stat-label">Removed</div>
                </div>
            </div>
            <button class="btn-restart" onclick="resetToBeginning()">Review Dataset Again</button>
        </div>
    </div>

    <!-- Empty dataset view -->
    <div id="empty-view" style="display: none; flex: 1; justify-content: center; align-items: center;">
        <div class="completion-container">
            <div class="completion-icon">📂</div>
            <h2 class="completion-title">No Images Found</h2>
            <p class="completion-desc">
                There are no image files located inside <span style="font-family: monospace; color: white;">dataset/peas</span>.<br>
                Please verify that the directory has image files (.jpg, .png, etc.) and restart the app.
            </p>
        </div>
    </div>

    <script>
        let localState = {
            masterList: [],
            currentIndex: 0,
            kept: [],
            history: []
        };

        let actionInProgress = false;

        async function fetchState() {
            try {
                const response = await fetch('/api/state');
                const data = await response.json();
                localState = data;
                updateUI();
            } catch (err) {
                console.error("Error fetching state:", err);
            }
        }

        function updateUI() {
            const list = localState.masterList;
            const idx = localState.currentIndex;
            const history = localState.history;
            const kept = localState.kept;

            // Handle empty states
            if (!list || list.length === 0) {
                document.getElementById('active-app-view').style.display = 'none';
                document.getElementById('completion-view').style.display = 'none';
                document.getElementById('empty-view').style.display = 'flex';
                return;
            }

            // Check if completed
            if (idx >= list.length) {
                document.getElementById('active-app-view').style.display = 'none';
                document.getElementById('completion-view').style.display = 'flex';
                document.getElementById('empty-view').style.display = 'none';
                
                document.getElementById('complete-total').innerText = list.length;
                document.getElementById('complete-keep').innerText = kept.length;
                document.getElementById('complete-remove').innerText = list.length - kept.length;
                return;
            }

            // Active application display
            document.getElementById('active-app-view').style.display = 'flex';
            document.getElementById('completion-view').style.display = 'none';
            document.getElementById('empty-view').style.display = 'none';

            // Show current image details
            const filename = list[idx];
            
            // Check if this image was removed in history previously (normally should not be active, but safe guard)
            const isRemoved = history.some(h => h.filename === filename && h.action === 'remove');
            const imagePath = isRemoved ? `/dataset/peas_removed/${encodeURIComponent(filename)}` : `/dataset/peas/${encodeURIComponent(filename)}`;
            
            const imgElement = document.getElementById('display-img');
            
            // Avoid visual jump if source is the same
            if (imgElement.getAttribute('src') !== imagePath) {
                imgElement.style.opacity = '0';
                imgElement.style.transform = 'scale(0.98)';
                
                // Set path and fade in once loaded
                setTimeout(() => {
                    imgElement.src = imagePath;
                }, 50);
            }

            document.getElementById('display-filename').innerText = filename;
            document.getElementById('display-filename').title = filename;
            document.getElementById('display-index').innerText = `Image ${idx + 1} of ${list.length}`;

            // Stats
            const total = list.length;
            const removedCount = history.filter(h => h.action === 'remove').length;
            const keptCount = kept.length;
            const processedCount = history.length;

            document.getElementById('stat-total').innerText = total;
            document.getElementById('stat-processed').innerText = processedCount;
            document.getElementById('stat-keep').innerText = keptCount;
            document.getElementById('stat-remove').innerText = removedCount;

            // Undo Button state
            document.getElementById('btn-undo-control').disabled = history.length === 0;

            // Progress bar
            const percent = total > 0 ? Math.round((processedCount / total) * 100) : 0;
            document.getElementById('app-progress-bar').style.width = `${percent}%`;
            document.getElementById('progress-percent-label').innerText = `${percent}%`;

            // History log
            const historyContainer = document.getElementById('history-container');
            historyContainer.innerHTML = '';
            document.getElementById('history-count').innerText = `(${history.length})`;

            // Display last 10 actions, reversed so latest is top
            const recentHistory = [...history].reverse().slice(0, 10);
            if (recentHistory.length === 0) {
                historyContainer.innerHTML = `
                    <div class="empty-warning">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <span style="font-size: 0.8rem;">No actions taken yet</span>
                    </div>`;
            } else {
                recentHistory.forEach(item => {
                    const el = document.createElement('div');
                    el.className = 'history-item';
                    
                    const dotClass = item.action === 'keep' ? 'keep' : 'remove';
                    const actionLabel = item.action === 'keep' ? 'Kept' : 'Removed';
                    const actionClass = item.action === 'keep' ? 'keep' : 'remove';
                    
                    el.innerHTML = `
                        <div class="history-info">
                            <div class="history-dot ${dotClass}"></div>
                            <span class="history-name" title="${item.filename}">${item.filename}</span>
                        </div>
                        <span class="history-action ${actionClass}">${actionLabel}</span>
                    `;
                    historyContainer.appendChild(el);
                });
            }

            // Preload next 3 images
            for (let i = 1; i <= 3; i++) {
                if (idx + i < list.length) {
                    const preloadName = list[idx + i];
                    const preloadImg = new Image();
                    preloadImg.src = `/dataset/peas/${encodeURIComponent(preloadName)}`;
                }
            }
        }

        // Image fade-in logic once loaded
        document.getElementById('display-img').onload = function() {
            this.style.opacity = '1';
            this.style.transform = 'scale(1)';
        };

        async function triggerAction(action) {
            if (actionInProgress) return;
            actionInProgress = true;

            const card = document.getElementById('main-image-card');
            
            // Visual feedback animation
            if (action === 'keep') {
                card.classList.add('flash-keep');
                setTimeout(() => card.classList.remove('flash-keep'), 200);
            } else if (action === 'remove') {
                card.classList.add('flash-remove');
                setTimeout(() => card.classList.remove('flash-remove'), 200);
            }

            try {
                const response = await fetch('/api/action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: action })
                });
                const data = await response.json();
                localState = data;
                updateUI();
            } catch (err) {
                console.error("Action error:", err);
            } finally {
                actionInProgress = false;
            }
        }

        async function resetToBeginning() {
            if (!confirm("Are you sure you want to review the dataset again? Your previous logs will be cleared, but files in 'peas_removed' will remain there until you manually restore them.")) {
                return;
            }
            try {
                // To reset, we tell the backend to clear history logs and reset index to 0
                // We'll write a simple hack: a specialized action 'reset'
                const response = await fetch('/api/action', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ action: 'undo-all' }) // we will support reset in backend, or undo-all
                });
                window.location.reload();
            } catch (err) {
                console.error(err);
            }
        }

        // Key bindings
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();
            
            if (key === 'k' || e.key === 'ArrowRight') {
                triggerAction('keep');
            } else if (key === 'r' || e.key === 'ArrowLeft' || e.key === 'Delete') {
                triggerAction('remove');
            } else if (key === 'z' || e.key === 'Backspace') {
                triggerAction('undo');
            }
        });

        // Initialize
        fetchState();
    </script>
</body>
</html>
"""

class CleanerHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Silence console flooding, log to console only important actions
        pass

    def do_GET(self):
        global current_index, master_list, kept_set, history
        
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
            
        elif path == "/api/state":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            res = {
                "masterList": master_list,
                "currentIndex": current_index,
                "kept": list(kept_set),
                "history": history
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
            
        elif path.startswith("/dataset/peas/"):
            filename = urllib.parse.unquote(path.replace("/dataset/peas/", ""))
            filepath = os.path.join(DATASET_DIR, filename)
            self.serve_file(filepath)
            
        elif path.startswith("/dataset/peas_removed/"):
            filename = urllib.parse.unquote(path.replace("/dataset/peas_removed/", ""))
            filepath = os.path.join(REMOVED_DIR, filename)
            self.serve_file(filepath)
            
        else:
            self.send_error(404, "File Not Found")
            
    def serve_file(self, filepath):
        if os.path.exists(filepath) and os.path.isfile(filepath):
            self.send_response(200)
            if filepath.lower().endswith(".jpg") or filepath.lower().endswith(".jpeg"):
                self.send_header("Content-Type", "image/jpeg")
            elif filepath.lower().endswith(".png"):
                self.send_header("Content-Type", "image/png")
            elif filepath.lower().endswith(".webp"):
                self.send_header("Content-Type", "image/webp")
            elif filepath.lower().endswith(".gif"):
                self.send_header("Content-Type", "image/gif")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            
            self.send_header("Cache-Control", "public, max-age=31536000")
            stat = os.stat(filepath)
            self.send_header("Content-Length", str(stat.st_size))
            self.end_headers()
            
            with open(filepath, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        global current_index, master_list, kept_set, history
        
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        
        if path == "/api/action":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            action = data.get("action")
            
            if action == "keep":
                if current_index < len(master_list):
                    filename = master_list[current_index]
                    kept_set.add(filename)
                    history.append({
                        "action": "keep",
                        "filename": filename,
                        "index": current_index
                    })
                    current_index += 1
                    save_state()
                    print(f"-> Kept: {filename} ({current_index}/{len(master_list)})")
                    
            elif action == "remove":
                if current_index < len(master_list):
                    filename = master_list[current_index]
                    
                    src = os.path.join(DATASET_DIR, filename)
                    dest = os.path.join(REMOVED_DIR, filename)
                    
                    if os.path.exists(src):
                        shutil.move(src, dest)
                        
                    history.append({
                        "action": "remove",
                        "filename": filename,
                        "index": current_index
                    })
                    current_index += 1
                    save_state()
                    print(f"-> Removed: {filename} ({current_index}/{len(master_list)})")
                    
            elif action == "undo":
                if len(history) > 0:
                    last_action = history.pop()
                    last_index = last_action["index"]
                    filename = last_action["filename"]
                    
                    if last_action["action"] == "remove":
                        src = os.path.join(REMOVED_DIR, filename)
                        dest = os.path.join(DATASET_DIR, filename)
                        if os.path.exists(src):
                            shutil.move(src, dest)
                            
                    elif last_action["action"] == "keep":
                        if filename in kept_set:
                            kept_set.remove(filename)
                            
                    current_index = last_index
                    save_state()
                    print(f"<- Undid last action, reverted back to: {filename} (Index {current_index})")
                    
            elif action == "undo-all":
                # Clear all histories and reset index, but keep files in peas_removed
                # This lets them restart the review session
                current_index = 0
                kept_set.clear()
                history.clear()
                save_state()
                print("<- Reset entire cleaning session index to 0.")
            
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            res = {
                "masterList": master_list,
                "currentIndex": current_index,
                "kept": list(kept_set),
                "history": history
            }
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_error(404, "Not Found")

def open_browser():
    time.sleep(0.8)
    url = f"http://localhost:{PORT}"
    print(f"Opening browser at: {url}")
    webbrowser.open(url)

if __name__ == "__main__":
    init_state()
    
    server_started = False
    while not server_started and PORT < 5100:
        try:
            handler = CleanerHTTPRequestHandler
            socketserver.TCPServer.allow_reuse_address = True
            with socketserver.TCPServer(("", PORT), handler) as httpd:
                print(f"==================================================")
                print(f"🚀 IMAGE CLEANER SERVER STARTED SUCCESSFULLY")
                print(f"📂 Dataset Folder: {os.path.abspath(DATASET_DIR)}")
                print(f"🗑️ Isolated Folder: {os.path.abspath(REMOVED_DIR)}")
                print(f"📊 Total Dataset Size: {len(master_list)} images")
                print(f"🌐 Server Url: http://localhost:{PORT}")
                print(f"⌨️ Controls: [K] or [→] Keep | [R] or [←] Remove | [Z] Undo")
                print(f"==================================================")
                
                threading.Thread(target=open_browser, daemon=True).start()
                
                server_started = True
                httpd.serve_forever()
        except OSError as e:
            if e.errno == 48 or "Address already in use" in str(e):
                PORT += 1
            else:
                raise e
