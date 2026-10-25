from pathlib import Path

from continualguard.attack_family_data import prepare


if __name__ == "__main__":
    print(prepare(Path(__file__).resolve().parents[1]))
