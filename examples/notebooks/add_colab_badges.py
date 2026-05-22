#!/usr/bin/env python3
"""Add Colab badge + conditional install cell to all notebooks."""
import json, os, glob

notebooks_dir = os.path.expanduser("~/StereoComplex/examples/notebooks")

COLAB_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "<table align=\"left\"><td>\n",
        "  <a target=\"_blank\" href=\"https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/{name}\">\n",
        "    <img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/>\n",
        "  </a>\n",
        "</td><td>\n",
        "  ⚠️ <b>Google Colab requires one extra setup cell before running this notebook.</b><br>\n",
        "  Run the cell immediately below, then proceed normally.\n",
        "</td></table>\n"
    ]
}

INSTALL_CELL = {
    "cell_type": "code",
    "metadata": {},
    "source": [
        "# ═══════════════════════════════════════════════\n",
        "# GOOGLE COLAB ONLY — skip this cell on your own machine\n",
        "# ═══════════════════════════════════════════════\n",
        "import sys, os\n",
        "IN_COLAB = \"google.colab\" in sys.modules\n",
        "if IN_COLAB:\n",
        "    !git clone https://github.com/jeffwitz/StereoComplex.git\n",
        "    %cd StereoComplex\n",
        "    !pip install -e \".[dev]\" 2>&1 | tail -3\n",
        "    os.chdir(\"examples/notebooks\")\n",
        "else:\n",
        "    print(\"Running locally — no action needed.\")\n"
    ],
    "execution_count": None,
    "outputs": []
}

for nb_path in sorted(glob.glob(f"{notebooks_dir}/*.ipynb")):
    with open(nb_path) as f:
        nb = json.load(f)
    
    name = os.path.basename(nb_path)
    badge_source = [
        f"<table align=\"left\"><td>\n",
        f"  <a target=\"_blank\" href=\"https://colab.research.google.com/github/jeffwitz/StereoComplex/blob/develop/examples/notebooks/{name}\">\n",
        f"    <img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/>\n",
        f"  </a>\n",
        f"</td><td>\n",
        f"  ⚠️ <b>Google Colab requires one extra setup cell before running this notebook.</b><br>\n",
        f"  Run the cell immediately below, then proceed normally.\n",
        f"</td></table>\n"
    ]
    
    badge_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": badge_source
    }
    install_code = {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ═══════════════════════════════════════════════\n",
            "# GOOGLE COLAB ONLY — skip this cell on your own machine\n",
            "# ═══════════════════════════════════════════════\n",
            "import sys, os\n",
            "IN_COLAB = \"google.colab\" in sys.modules\n",
            "if IN_COLAB:\n",
            "    !git clone https://github.com/jeffwitz/StereoComplex.git\n",
            "    %cd StereoComplex\n",
            "    !pip install -e \".[dev]\" 2>&1 | tail -3\n",
            "    os.chdir(\"examples/notebooks\")\n",
            "else:\n",
            "    print(\"Running locally — no action needed.\")\n"
        ],
        "execution_count": None,
        "outputs": []
    }
    
    # Insert at top (after any initial markdown title)
    cells = nb["cells"]
    # Find first cell — insert badge + install after it
    # Actually, insert at position 0 (very top)
    cells.insert(0, install_code)
    cells.insert(0, badge_md)
    
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  ✓ {name}")
