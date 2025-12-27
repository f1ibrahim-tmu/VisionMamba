#!/bin/bash
# Script to verify which commits from custom_cuda are unique vs duplicates in mmcv-2.x

echo "=== Commits in custom_cuda NOT in mmcv-2.x ==="
echo ""

# Commits that need verification (similar messages exist in both branches)
declare -A verify_commits=(
    ["932dc6d"]="6f867f6"  # mamba-ssm path fix
    ["cdda7d0"]="d9fd21a"  # selective_scan_cuda.fwd() fix
    ["9a16cee"]="944069c"  # RK4 fix
    ["2d50daa"]="7866632"  # bilinear fix
    ["3d7b19e"]="310facf"  # verification script
    ["e962cb7"]="01c76bc"  # benchmarking scripts
    ["19fb2bc"]="ddd5a1f"  # ZeroDivisionError fix
    ["644dac1"]="c23f2ae"  # Formula fixes
    ["1e10a2b"]="c23f2ae"  # FOH formula fix
)

echo "Checking commits that may be duplicates..."
echo ""

for custom_commit in "${!verify_commits[@]}"; do
    mmcv_commit="${verify_commits[$custom_commit]}"
    echo "--- Comparing $custom_commit (custom_cuda) vs $mmcv_commit (mmcv-2.x) ---"
    
    # Get commit messages
    custom_msg=$(git log -1 --format="%s" $custom_commit)
    mmcv_msg=$(git log -1 --format="%s" $mmcv_commit)
    
    echo "Custom: $custom_msg"
    echo "MMCV:   $mmcv_msg"
    
    # Check if files changed are similar
    custom_files=$(git show --name-only --format="" $custom_commit | sort)
    mmcv_files=$(git show --name-only --format="" $mmcv_commit | sort)
    
    if [ "$custom_files" == "$mmcv_files" ]; then
        echo "✅ Same files changed"
    else
        echo "⚠️  Different files changed"
        echo "Custom files:"
        echo "$custom_files" | head -5
        echo "..."
        echo "MMCV files:"
        echo "$mmcv_files" | head -5
        echo "..."
    fi
    
    # Check if commits are actually different
    if git diff $custom_commit $mmcv_commit --quiet; then
        echo "✅ Commits are identical (same content, different hash)"
    else
        echo "⚠️  Commits are different - review needed"
        echo "   Run: git diff $custom_commit $mmcv_commit --stat"
    fi
    
    echo ""
done

echo ""
echo "=== Unique commits (definitely not in mmcv-2.x) ==="
echo ""

unique_commits=(
    "ebf2bdc"  # Add custom CUDA kernels
    "0608edb"  # Add CUDA/Python selection
    "4774b22"  # Fix complex type template
    "fb13860"  # Non-causal bidirectional scan
    "a429ac3"  # Diagnostic script
    "51e7c79"  # HPC-specific config
)

for commit in "${unique_commits[@]}"; do
    msg=$(git log -1 --format="%s" $commit)
    echo "✅ $commit - $msg"
done

echo ""
echo "=== Summary ==="
echo "Run 'git show <commit>' to see full details of any commit"
echo "Run 'git diff <custom_commit> <mmcv_commit>' to compare two commits"

