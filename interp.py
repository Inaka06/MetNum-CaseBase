"""CB-03 Interpolasi RSSI nyata UJIIndoorLoc. Jalankan dari folder berkas ini."""
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent)
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import urllib.request
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, BarycentricInterpolator, CloughTocher2DInterpolator, LinearNDInterpolator

DATA = Path("data")
OUT = Path("hasil")
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.grid": True})
np.set_printoptions(precision=6, suppress=True)
print("Python", platform.python_version())
print("NumPy", np.__version__, "pandas", pd.__version__, "SciPy", scipy.__version__)

url = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
archive = DATA / "ujiindoorloc.zip"
local_csv = DATA / "trainingData.csv"
if local_csv.exists():
    raw = pd.read_csv(local_csv)
    source_hash = hashlib.sha256(local_csv.read_bytes()).hexdigest()
else:
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            archive.write_bytes(response.read())
    source_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as z:
        name = next(n for n in z.namelist() if n.endswith("trainingData.csv"))
        raw = pd.read_csv(BytesIO(z.read(name)))
wap_cols = sorted(c for c in raw if c.startswith("WAP"))
assert len(wap_cols) == 520
print("Ukuran data:", raw.shape)
print("SHA256 sumber:", source_hash)
print(raw.groupby(["BUILDINGID", "FLOOR"]).size().to_string())

counts = raw.groupby(["BUILDINGID", "FLOOR"]).size().sort_values(ascending=False)
building, floor = map(int, counts.index[0])
subset = raw[(raw.BUILDINGID == building) & (raw.FLOOR == floor)].copy()
subset[wap_cols] = subset[wap_cols].replace(100, np.nan)
locations = subset.groupby(["LONGITUDE", "LATITUDE"], sort=True)[wap_cols].median().reset_index()
xy = locations[["LONGITUDE", "LATITUDE"]].to_numpy()
origin = xy.mean(axis=0)
centered = xy - origin
best_count = -1
for angle in range(180):
    theta = np.deg2rad(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    perpendicular = np.array([-direction[1], direction[0]])
    t = centered @ direction
    q = centered @ perpendicular
    for center in q:
        mask = np.abs(q - center) <= 1.0
        if mask.sum() > best_count:
            best_count = int(mask.sum())
            best = (angle, float(center), direction.copy(), perpendicular.copy(), t.copy(), q.copy(), mask.copy())
angle, center, direction, perpendicular, t, q, mask = best
strip = locations.loc[mask].copy()
strip["t"] = t[mask]
strip["offset"] = q[mask] - center
strip = strip.sort_values("t")
strip["cluster"] = (strip.t.diff().fillna(np.inf) > 2.0).cumsum()
indices = strip.offset.abs().groupby(strip.cluster).idxmin()
representatives = strip.loc[indices].sort_values("t")
coverage = representatives[wap_cols].notna().sum()
ap = sorted(wap_cols, key=lambda c: (-int(coverage[c]), c))[0]
profile = representatives.dropna(subset=[ap]).copy()
profile["s"] = profile.t - profile.t.min()
profile["RSSI"] = profile[ap]
profile = profile[["s", "RSSI", "LONGITUDE", "LATITUDE", "offset"]].reset_index(drop=True)
assert len(profile) >= 9, "Penampang terlalu sedikit untuk rancangan uji ini."
assert np.all(np.diff(profile.s) > 0)
print(f"Gedung {building}, lantai {floor}, AP {ap}")
print(f"Baris lantai: {len(subset)}, koordinat unik: {len(locations)}")
print(f"Sudut {angle} derajat, titik dalam pita {best_count}, titik wakil {len(profile)}")
print(profile.round(4).to_string(index=True))
profile.to_csv(OUT / "titik_penampang.csv", index=False)

n_test = max(3, round(0.3 * len(profile)))
test_indices = np.unique(np.rint(np.linspace(2, len(profile) - 3, n_test)).astype(int))
is_test = np.isin(np.arange(len(profile)), test_indices)
profile["peran"] = np.where(is_test, "uji", "simpul")
train = profile.loc[~is_test]
test = profile.loc[is_test]
x = train.s.to_numpy()
y = train.RSSI.to_numpy()
x_test = test.s.to_numpy()
y_test = test.RSSI.to_numpy()
grid = np.linspace(x[0], x[-1], 1501)
print(profile[["s", "RSSI", "peran"]].round(4).to_string(index=False))
print("Simpul:", len(train), "Lokasi uji:", len(test), "Derajat Lagrange:", len(train) - 1)
profile.to_csv(OUT / "titik_penampang.csv", index=False)

def check_nodes(xn, yn):
    xn, yn = np.asarray(xn, float), np.asarray(yn, float)
    if len(xn) < 2 or xn.ndim != 1 or yn.shape != xn.shape:
        raise ValueError("Diperlukan minimal dua pasangan simpul 1D.")
    if not np.all(np.isfinite(xn)) or not np.all(np.isfinite(yn)) or np.any(np.diff(xn) <= 0):
        raise ValueError("Simpul harus finite, unik, dan berurutan naik.")
    return xn, yn


def lagrange(xn, yn, xq):
    xn, yn = check_nodes(xn, yn)
    xq = np.atleast_1d(xq).astype(float)
    midpoint, scale = (xn[0] + xn[-1]) / 2, (xn[-1] - xn[0]) / 2
    nodes = (xn - midpoint) / scale
    queries = (xq - midpoint) / scale
    hasil = np.zeros_like(queries)
    for i in range(len(nodes)):
        basis = np.ones_like(queries)
        for j in range(len(nodes)):
            if j != i:
                basis *= (queries - nodes[j]) / (nodes[i] - nodes[j])
        hasil += yn[i] * basis
    return hasil

def spline_coef(xn, yn):
    xn, yn = check_nodes(xn, yn)
    n = len(xn)
    h = np.diff(xn)
    A = np.zeros((n, n))
    rhs = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6 * ((yn[i + 1] - yn[i]) / h[i] - (yn[i] - yn[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def spline_eval(xn, yn, M, xq):
    xn, yn = check_nodes(xn, yn)
    xq = np.atleast_1d(xq).astype(float)
    if np.any((xq < xn[0]) | (xq > xn[-1])):
        raise ValueError("Pertanyaan di luar rentang interpolasi.")
    i = np.clip(np.searchsorted(xn, xq, side="right") - 1, 0, len(xn) - 2)
    h = xn[i + 1] - xn[i]
    a, b = xn[i + 1] - xq, xq - xn[i]
    return (M[i] * a**3 / (6*h) + M[i+1] * b**3 / (6*h)
            + (yn[i]/h - M[i]*h/6)*a + (yn[i+1]/h - M[i+1]*h/6)*b)


M = spline_coef(x, y)
print("Turunan kedua pada simpul:", M)

cs = CubicSpline(x, y, bc_type="natural", extrapolate=False)
bi = BarycentricInterpolator(x, y)
verification = {
    "Lagrange manual vs SciPy": float(np.max(np.abs(lagrange(x, y, grid) - bi(grid)))),
    "Spline manual vs SciPy": float(np.max(np.abs(spline_eval(x, y, M, grid) - cs(grid)))),
    "Residual simpul Lagrange": float(np.max(np.abs(lagrange(x, y, x) - y))),
    "Residual simpul spline": float(np.max(np.abs(spline_eval(x, y, M, x) - y))),
}
for name, value in verification.items():
    print(f"{name}: {value:.3e}")
    assert value < 1e-7
assert np.allclose(M[[0, -1]], 0, atol=1e-10)
assert np.allclose(M, cs(x, 2), atol=1e-8)
for xn, yn in [(np.array([0., 0.7, 2., 3.5]), np.array([1., -2., 0., 4.])),
               (np.array([0., 2.]), np.array([1., 5.]))]:
    query = np.linspace(xn[0], xn[-1], 101)
    assert np.allclose(spline_eval(xn, yn, spline_coef(xn, yn), query),
                       CubicSpline(xn, yn, bc_type="natural")(query), atol=1e-10)
print("Semua pemeriksaan numerik lulus.")

predictions = {
    "Lagrange": lagrange(x, y, x_test),
    "Natural Cubic Spline": spline_eval(x, y, M, x_test),
    "Linear": np.interp(x_test, x, y),
}
estimates = pd.DataFrame({"s": x_test, "RSSI_rujukan": y_test, **predictions})
rows = []
for method, pred in predictions.items():
    error = pred - y_test
    rows.append({"metode": method, "MAE_dB": np.mean(np.abs(error)),
                 "RMSE_dB": np.sqrt(np.mean(error**2)), "maks_galat_dB": np.max(np.abs(error))})
metrics = pd.DataFrame(rows)
print("Estimasi RSSI (dBm):")
print(estimates.round(4).to_string(index=False))
print("\nGalat pada lokasi uji (dB):")
print(metrics.round(4).to_string(index=False))
estimates.to_csv(OUT / "estimasi_uji.csv", index=False)
metrics.to_csv(OUT / "metrik_uji.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(grid, lagrange(x, y, grid), label=f"Lagrange derajat {len(x)-1}", color="tab:red")
ax.plot(grid, spline_eval(x, y, M, grid), label="Natural Cubic Spline", color="tab:blue")
ax.plot(grid, np.interp(grid, x, y), "--", label="Linear", color="0.5")
ax.scatter(x, y, c="black", label="Simpul", zorder=4)
ax.scatter(x_test, y_test, marker="x", s=85, c="tab:green", label="Lokasi uji", zorder=5)
ax.set(xlabel="Posisi sepanjang penampang (satuan proyeksi)", ylabel="RSSI (dBm)", title="Estimasi RSSI pada penampang nyata")
ax.legend(ncol=2)
fig.tight_layout()
fig.savefig(OUT / "01_interpolasi_rssi.png", dpi=180)
plt.show()

sensitivity_rows = []
for k in sorted(set([4, 6, len(x)])):
    selected = np.unique(np.rint(np.linspace(0, len(x)-1, k)).astype(int))
    xn, yn = x[selected], y[selected]
    for method, pred in {
        "Lagrange": lagrange(xn, yn, x_test),
        "Natural Cubic Spline": spline_eval(xn, yn, spline_coef(xn, yn), x_test)
    }.items():
        sensitivity_rows.append({"simpul": len(xn), "derajat_global": len(xn)-1,
                                 "metode": method, "RMSE_dB": np.sqrt(np.mean((pred-y_test)**2))})
sensitivity = pd.DataFrame(sensitivity_rows)
print(sensitivity.round(4).to_string(index=False))
sensitivity.to_csv(OUT / "sensitivitas_simpul.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
for method, g in sensitivity.groupby("metode", sort=False):
    ax.plot(g.simpul, g.RMSE_dB, "o-", label=method)
ax.set(xlabel="Jumlah simpul", ylabel="RMSE uji (dB)", title="Sensitivitas pada lokasi uji yang sama")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "02_sensitivitas.png", dpi=180)
plt.show()

def runge(u):
    return 1 / (1 + 25 * u**2)

u = np.linspace(-1, 1, 4001)
truth = runge(u)
runge_rows = []
for degree in [4, 8, 12, 16, 20]:
    uniform = np.linspace(-1, 1, degree+1)
    cheb = np.sort(np.cos(np.arange(degree+1) * np.pi / degree))
    values = runge(uniform)
    curves = {
        "Lagrange seragam": lagrange(uniform, values, u),
        "Natural Cubic Spline": spline_eval(uniform, values, spline_coef(uniform, values), u),
        "Lagrange Chebyshev": lagrange(cheb, runge(cheb), u),
    }
    # Bandingkan dengan SciPy untuk mengecek perhitungan Lagrange.
    assert np.allclose(curves["Lagrange seragam"], BarycentricInterpolator(uniform, values)(u), atol=1e-7, rtol=1e-8)
    for method, curve in curves.items():
        runge_rows.append({"derajat": degree, "metode": method, "maks_galat_grid": np.max(np.abs(curve-truth))})
runge_metrics = pd.DataFrame(runge_rows)
print(runge_metrics.round(6).to_string(index=False))
runge_metrics.to_csv(OUT / "runge.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(u, truth, "k--", label="Fungsi Runge")
for method, curve in curves.items():
    axes[0].plot(u, curve, label=method)
axes[0].scatter(uniform, values, s=12, c="k")
axes[0].set(title="Derajat 20 tanpa pemotongan sumbu", xlabel="u", ylabel="f(u)")
axes[0].legend(fontsize=8)
for method, g in runge_metrics.groupby("metode", sort=False):
    axes[1].semilogy(g.derajat, g.maks_galat_grid, "o-", label=method)
axes[1].set(title="Galat maksimum pada 4001 titik", xlabel="Derajat global / jumlah simpul - 1", ylabel="Galat absolut")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "03_runge.png", dpi=180)
plt.show()

spatial = locations.dropna(subset=[ap]).copy()
coords = spatial[["LONGITUDE", "LATITUDE"]].to_numpy() - origin
values2d = spatial[ap].to_numpy()
gx, gy = np.meshgrid(np.linspace(coords[:,0].min(), coords[:,0].max(), 180),
                     np.linspace(coords[:,1].min(), coords[:,1].max(), 180))
interp2d = LinearNDInterpolator(coords, values2d, fill_value=np.nan)
gz = interp2d(gx, gy)
cubic2d = CloughTocher2DInterpolator(coords, values2d, fill_value=np.nan)(gx, gy)
contour_ranges = {"RSSI terukur": [float(values2d.min()), float(values2d.max())],
                  "Kontur linear": [float(np.nanmin(gz)), float(np.nanmax(gz))],
                  "Kubik 2D diagnostik": [float(np.nanmin(cubic2d)), float(np.nanmax(cubic2d))]}
print("Rentang nilai (dBm):", contour_ranges)
fig, ax = plt.subplots(figsize=(8.5, 5.5))
levels = np.linspace(np.nanmin(gz), np.nanmax(gz), 18)
contour = ax.contourf(gx, gy, np.ma.masked_invalid(gz), levels=levels, cmap="viridis")
ax.scatter(coords[:,0], coords[:,1], s=10, c="black", alpha=0.6, label="Koordinat terukur")
pp = profile[["LONGITUDE", "LATITUDE"]].to_numpy() - origin
ax.plot(pp[:,0], pp[:,1], "r.-", label="Penampang 1D")
ax.set(xlabel="X relatif (satuan proyeksi)", ylabel="Y relatif (satuan proyeksi)",
       title=f"Kontur linear RSSI {ap} | Gedung {building}, lantai {floor}")
ax.set_aspect("equal")
ax.legend(loc="best", fontsize=8)
fig.colorbar(contour, ax=ax, label="Estimasi RSSI (dBm)")
fig.tight_layout()
fig.savefig(OUT / "04_kontur.png", dpi=180)
plt.show()

best_method = metrics.sort_values("RMSE_dB").iloc[0]
summary = {
    "building": building, "floor": floor, "ap": ap, "rows_floor": len(subset),
    "unique_locations_floor": len(locations), "spatial_locations": len(spatial),
    "angle": angle, "strip_center": center, "strip_points": best_count,
    "profile_points": len(profile), "train_points": len(train), "test_points": len(test),
    "span": float(profile.s.max()), "max_offset": float(profile.offset.abs().max()),
    "source_sha256": source_hash, "verification": verification, "contour_ranges": contour_ranges,
    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                 "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
    "metrics": metrics.to_dict(orient="records"),
}
(OUT / "ringkasan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(f"RMSE terendah pada uji ini: {best_method.metode} = {best_method.RMSE_dB:.4f} dB")
print("Simpan kesimpulan sesuai angka aktual, bukan asumsi bahwa spline selalu menang.")
print("Tabel, gambar, dan metadata eksekusi tersimpan pada folder hasil.")
