import sys

def parse_matrices(text):
    blocks = [b.strip() for b in text.strip().split('\n\n') if b.strip()]
    matrices = []
    for block in blocks:
        rows = []
        for line in block.strip().split('\n'):
            rows.append([float(x) for x in line.split()])
        matrices.append(rows)
    return matrices

def matmul(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"Incompatible dimensions: {cols_A} != {rows_B}"
    C = [[sum(A[i][k] * B[k][j] for k in range(cols_A)) for j in range(cols_B)] for i in range(rows_A)]
    return C

def matrices_close(C_expected, C_given, rel_tol=1e-2):
    if len(C_expected) != len(C_given):
        return False, 0, "Row count mismatch"

    error_count = 0
    first_mismatch = None

    for i, (row_e, row_g) in enumerate(zip(C_expected, C_given)):
        if len(row_e) != len(row_g):
            return False, 0, f"Column count mismatch at row {i}"
        for j, (e, g) in enumerate(zip(row_e, row_g)):
            diff = abs(e - g)
            rel = diff / max(1.0, abs(e))
            if rel > rel_tol:
                error_count += 1
                if first_mismatch is None:
                    first_mismatch = (
                        f"Mismatch at [{i}][{j}]: expected {e:.5f}, got {g:.5f}, "
                        f"diff={diff:.6f}, rel={rel:.6e}"
                    )

    if error_count > 0:
        return False, error_count, first_mismatch
    return True, 0, "OK"

data = sys.stdin.read().replace('\r\n', '\n').replace('\r', '\n')
matrices = parse_matrices(data)

if len(matrices) != 3:
    print(f"ERROR: Expected 3 matrices, got {len(matrices)}")
    sys.exit(1)

A, B, C_given = matrices
C_expected = matmul(A, B)

ok, error_count, msg = matrices_close(C_expected, C_given)
if ok:
    print("CORRECT")
else:
    print(f"WRONG: errors={error_count}; {msg}")
