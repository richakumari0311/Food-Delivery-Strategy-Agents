#!/bin/bash
# Run from project root: bash fix_structure.sh
# Fixes the clear-cut issues found. Does NOT touch src/config.py, src/db.py,
# src/embeddings.py - check those manually first (see step 5 below).

set -e  # stop on first error so you can see exactly what happened

echo "1. Moving misplaced agent files from src/utils/ to src/agents/ ..."
mv src/utils/review_analysis.py src/agents/review_analysis.py
mv src/utils/segmentation.py src/agents/segmentation.py
mv src/utils/strategy.py src/agents/strategy.py

echo "2. Removing duplicate retry.py under src/agents/utils/ ..."
rm -rf src/agents/utils

echo "3. Moving .streamlit/ to project root ..."
mv app/.streamlit .streamlit

echo "4. Renaming sanity check script for consistency with README ..."
mv scripts/06.1_sanity_checks.py scripts/06b_sanity_checks.py

echo ""
echo "Done with the safe, clear-cut fixes."
echo ""
echo "5. Checking whether src/config.py, src/db.py, src/embeddings.py are used anywhere:"
grep -rn "from src.config\|from src\.db import\|from src.embeddings\|import src\.config\|import src\.db\b\|import src\.embeddings" . --include="*.py" 2>/dev/null || echo "   No references found in any .py file - these 3 files are likely unused."
echo ""
echo "   Line counts (to see if they're empty stubs or have real content):"
wc -l src/config.py src/db.py src/embeddings.py 2>/dev/null

echo ""
echo "Now verify nothing broke:"
python3 -c "import ast; [ast.parse(open(f).read()) for f in ['src/agents/review_analysis.py','src/agents/segmentation.py','src/agents/strategy.py','src/agents/data_analyst.py','src/agents/competitor_research.py','src/orchestrator.py']]" && echo "All agent files still parse correctly."