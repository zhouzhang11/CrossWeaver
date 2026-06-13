# 找到 mamba_ssm 的安装目录
PKG_DIR=$(python - <<'PY'
import mamba_ssm, os
print(os.path.dirname(mamba_ssm.__file__))
PY
)

echo "mamba-ssm installed at: $PKG_DIR"

# 1) 替换 modules 下的两个文件
install -m 644 ./ssm/mamba_simple.py "$PKG_DIR/modules/mamba_simple.py"
install -m 644 ./ssm/local_scan.py   "$PKG_DIR/modules/local_scan.py"

# 2) 替换 ops 下的一个文件
install -m 644 ./ssm/selective_scan_interface.py "$PKG_DIR/ops/selective_scan_interface.py"
