import json
import copy
import os
from colorama import Fore, Style, init

init(autoreset=True)

# ----------------- Pfade -----------------
INPUT_FILE = "Samples/SilverStandard_251201.json"
OUTPUT_FILE = "Samples/SilverStandard_Final.json"
STATE_FILE = "Samples/state.json"

# ----------------- Hilfsfunktionen -----------------
def load_state():
    """Lade letzte Position. Gibt 0 zurück, wenn Datei leer oder kaputt."""
    if os.path.exists(STATE_FILE):
        try:
            if os.path.getsize(STATE_FILE) == 0:
                return 0
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("index", 0)
        except (json.JSONDecodeError, ValueError):
            return 0
    return 0

def save_state(index):
    """Speichere aktuelle Position."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"index": index}, f, indent=2)

def safe_save_output(cleaned):
    """Sicheres Speichern in OUTPUT_FILE (Atomar)."""
    temp = OUTPUT_FILE + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=4, ensure_ascii=False)
    os.replace(temp, OUTPUT_FILE)

def pretty_print(entry, index, total):
    """Schöne farbige Ausgabe."""
    print("\n" + Fore.CYAN + "=" * 70)
    print(Fore.CYAN + f" Eintrag {index + 1} von {total} ".center(70, "="))
    print(Fore.CYAN + "=" * 70 + Style.RESET_ALL)
    for key, value in entry.items():
        print(Fore.YELLOW + f"{key:<15}: " + Style.RESET_ALL + f"{value}")
    print(Fore.CYAN + "-" * 70 + Style.RESET_ALL)

def edit_entry(entry):
    """Interaktives Editieren eines Eintrags."""
    print(Fore.GREEN + "\nEDIT MODE – ENTER drücken, um Wert zu behalten." + Style.RESET_ALL)
    updated = {}
    for key, value in entry.items():
        new_val = input(f"{Fore.YELLOW}{key}{Style.RESET_ALL} [{value}]: ").strip()
        updated[key] = new_val if new_val != "" else value
    return updated

def load_or_create_output():
    """Lädt OUTPUT_FILE oder startet mit leerer Liste."""
    if os.path.exists(OUTPUT_FILE):
        try:
            if os.path.getsize(OUTPUT_FILE) == 0:
                return []
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(Fore.RED + "⚠ Warnung: output.json leer oder beschädigt. Neu starten." + Style.RESET_ALL)
            return []
    return []

# ----------------- Hauptfunktion -----------------
def main():
    # Original-Daten laden
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    cleaned = load_or_create_output()
    start_index = load_state()

    print(Fore.GREEN + f"Loaded {total} entries. Resuming from index {start_index + 1}.\n" + Style.RESET_ALL)

    for idx in range(start_index, total):
        entry = data[idx]
        pretty_print(entry, idx, total)

        while True:
            cmd = input(Fore.CYAN + "(Y) Keep  (N) Delete  (E) Edit  (S) Skip  (Q) Quit → " + Style.RESET_ALL).strip().lower()

            if cmd == "y":
                cleaned.append(entry)
                safe_save_output(cleaned)
                print(Fore.GREEN + "✓ Kept and saved." + Style.RESET_ALL)
                break

            elif cmd == "n":
                print(Fore.RED + "✗ Deleted (not saved)." + Style.RESET_ALL)
                break

            elif cmd == "e":
                new_entry = edit_entry(entry)
                entry = new_entry
                data[idx] = new_entry   # editieren übernehmen
                pretty_print(entry, idx, total)

            elif cmd == "s":
                print(Fore.MAGENTA + "↷ Skipped." + Style.RESET_ALL)
                break

            elif cmd == "q":
                print(Fore.YELLOW + "Stopping and saving progress..." + Style.RESET_ALL)
                save_state(idx)
                safe_save_output(cleaned)
                return

            else:
                print(Fore.RED + "Invalid input. Use Y / N / E / S / Q." + Style.RESET_ALL)

        # Nach jedem Eintrag speichern
        save_state(idx + 1)
        safe_save_output(cleaned)

    print(Fore.GREEN + "\nAll entries processed!" + Style.RESET_ALL)
    save_state(total)
    safe_save_output(cleaned)

# ----------------- Entry Point -----------------
if __name__ == "__main__":
    main()
