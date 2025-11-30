# 星历计算_georinex.py
import numpy as np
import pandas as pd
import georinex as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

# ==============================
# 常量定义
# ==============================
GM = 3.986005e14  # 地球引力常数 (m^3/s^2)
OMEGA_E = 7.2921151467e-5  # 地球自转角速度 (rad/s)
GPS_EPOCH = datetime(1980, 1, 6)  # GPS 时间起点

def dt_to_sow(dt: datetime) -> float:
    """Convert GPS datetime to Seconds of Week (SOW)."""
    total_seconds = (dt - GPS_EPOCH).total_seconds()
    return total_seconds % 604800

# ==============================
# 辅助函数：解 Kepler 方程（仅用于 toe 时刻，t_k=0）
# ==============================
def solve_kepler(M, e, tol=1e-12, max_iter=10):
    E = M.copy() if hasattr(M, '__len__') else np.array([M])
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E[0] if np.isscalar(M) else E

# ==============================
# 计算卫星在 toe 时刻的位置（t_k = 0）
# ==============================
def compute_at_toe(eph):
    """
    eph: dict-like with broadcast ephemeris parameters at a given epoch.
    Returns position in kilometers (X, Y, Z).
    """
    sqrtA = eph['sqrtA']
    e = eph['Eccentricity']
    i0 = eph['Io']
    Omega0 = eph['Omega0']
    omega = eph['omega']
    M0 = eph['M0']
    Delta_n = eph['DeltaN']
    Cuc = eph['Cuc']
    Cus = eph['Cus']
    Cic = eph['Cic']
    Cis = eph['Cis']
    Crc = eph['Crc']
    Crs = eph['Crs']
    idot = eph['IDOT']
    Omegadot = eph['OmegaDot']
    toe_dt = eph['toe_datetime']  # already GPS time

    A = sqrtA ** 2
    n0 = np.sqrt(GM / (A ** 3))
    n = n0 + Delta_n
    tk = 0.0  # 因为我们在 toe 时刻
    M = M0 + n * tk  # = M0

    E = solve_kepler(M, e)
    sinv = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    cosv = (np.cos(E) - e) / (1 - e * np.cos(E))
    v = np.arctan2(sinv, cosv)

    u0 = v + omega
    r0 = A * (1 - e * np.cos(E))

    delta_u = Cuc * np.cos(2*u0) + Cus * np.sin(2*u0)
    delta_r = Crc * np.cos(2*u0) + Crs * np.sin(2*u0)
    delta_i = Cic * np.cos(2*u0) + Cis * np.sin(2*u0)

    u = u0 + delta_u
    r = r0 + delta_r
    i = i0 + delta_i + idot * tk  # tk=0

    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)

    # 注意：Omega 在 toe 时刻 = Omega0 - OMEGA_E * toe_sow
    toe_sow = dt_to_sow(toe_dt)
    Omega = Omega0 + (Omegadot - OMEGA_E) * tk - OMEGA_E * toe_sow  # tk=0 → -OMEGA_E * toe_sow

    X = x_prime * np.cos(Omega) - y_prime * np.cos(i) * np.sin(Omega)
    Y = x_prime * np.sin(Omega) + y_prime * np.cos(i) * np.cos(Omega)
    Z = y_prime * np.sin(i)

    return X / 1000.0, Y / 1000.0, Z / 1000.0  # km

# ==============================
# 解析 SP3 文件（假设时间为 GPS 时间）
# ==============================
def parse_sp3(file_path):
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        current_gps_dt = None
        for line in lines:
            if line.startswith('*'):
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
                hour, minute = int(parts[4]), int(parts[5])
                sec = float(parts[6])
                current_gps_dt = datetime(year, month, day, hour, minute, int(sec), int((sec - int(sec)) * 1e6))
            elif line.startswith('P') and len(line) > 50:
                sys_char = line[1]
                prn_str = line[2:4].strip()
                if sys_char == 'G' and prn_str.isdigit():
                    sv_id = f"G{int(prn_str):02d}"
                    try:
                        x = float(line[4:18]) * 1000  # km → m
                        y = float(line[18:32]) * 1000
                        z = float(line[32:46]) * 1000
                        if sv_id not in data:
                            data[sv_id] = {}
                        data[sv_id][current_gps_dt] = np.array([x, y, z])
                    except Exception:
                        continue
    return data

# ==============================
# 主程序
# ==============================
def main():
    brdc_file = "BRDC00GOP_R_20253220000_01D_MN.rnx"
    sp3_file = "WUM0MGXRAP_20253220000_01D_05M_ORB.SP3"

    os.makedirs("plots", exist_ok=True)
    os.makedirs("coordinates", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    print("Loading broadcast ephemeris with georinex...")
    nav = gr.load(brdc_file)
    gps_nav = nav.where(nav['sv'] != '', drop=True)
    gps_nav = gps_nav.sel(sv=[sv for sv in gps_nav.sv.values if sv.startswith('G')])

    # 构建广播星历记录列表（每个记录包含一个卫星在某个 Toe 的完整参数）
    df_list = []
    for sv in gps_nav.sv.values:
        ds = gps_nav.sel(sv=sv).dropna(dim='time', how='all')
        for t in ds.time.values:
            rec = ds.sel(time=t).to_pandas()
            if not rec.isnull().all():
                gps_dt = pd.Timestamp(t).to_pydatetime()  # georinex.time is GPS time
                rec_dict = rec.to_dict()
                rec_dict['sv'] = sv
                rec_dict['toe_datetime'] = gps_dt
                df_list.append(rec_dict)

    if not df_list:
        raise RuntimeError("No valid broadcast ephemeris found!")

    brdc_df = pd.DataFrame(df_list)
    print(f"Loaded {len(brdc_df)} broadcast ephemeris records from {brdc_df['sv'].nunique()} satellites.")

    print("Loading SP3 precise orbits...")
    sp3_data = parse_sp3(sp3_file)  # dict: Gxx -> {datetime: [x,y,z]}

    results = []
    satellite_errors = {}

    for _, row in brdc_df.iterrows():
        sv = row['sv']
        toe_dt = row['toe_datetime']

        # 精确匹配 SP3 时间（可扩展为 ±30 秒容差，此处保持严格匹配）
        if sv not in sp3_data or toe_dt not in sp3_data[sv]:
            continue

        try:
            x_calc, y_calc, z_calc = compute_at_toe(row)  # km
            x_sp3, y_sp3, z_sp3 = sp3_data[sv][toe_dt] / 1000.0  # convert m → km

            error_m = np.linalg.norm([x_calc - x_sp3, y_calc - y_sp3, z_calc - z_sp3]) * 1000  # meters

            results.append({
                'satellite': sv,
                'timestamp': toe_dt,
                'error_m': error_m,
                'X_calc_km': x_calc,
                'Y_calc_km': y_calc,
                'Z_calc_km': z_calc,
                'X_sp3_km': x_sp3,
                'Y_sp3_km': y_sp3,
                'Z_sp3_km': z_sp3
            })

            if sv not in satellite_errors:
                satellite_errors[sv] = {'times': [], 'errors': []}
            satellite_errors[sv]['times'].append(toe_dt)
            satellite_errors[sv]['errors'].append(error_m)

        except Exception as e:
            print(f"Error computing position for {sv} at {toe_dt}: {e}")
            continue

    if not results:
        raise RuntimeError("No matching epochs between broadcast toe and SP3!")

    result_df = pd.DataFrame(results)
    print(f"Successfully matched {len(result_df)} epochs.")

    # Save coordinates (broadcast only, at toe)
    coord_df = result_df[['satellite', 'timestamp', 'X_calc_km', 'Y_calc_km', 'Z_calc_km']].copy()
    for sv in coord_df['satellite'].unique():
        sv_coord = coord_df[coord_df['satellite'] == sv]
        sv_coord[['timestamp', 'X_calc_km', 'Y_calc_km', 'Z_calc_km']].to_csv(
            f"coordinates/{sv}_toe_coordinates.csv", index=False
        )

    # Plot per-satellite error
    for sv, data in satellite_errors.items():
        times = data['times']
        errors = data['errors']
        plt.figure(figsize=(10, 4))
        plt.plot(times, errors, 'o-', markersize=4)
        plt.xlabel('Time (Broadcast Ephemeris toe)')
        plt.ylabel('3D Error at toe (meters)')
        plt.title(f'{sv} Broadcast vs SP3 Error at toe')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"plots/{sv}_error.png", dpi=150)
        plt.close()

    # Overall plot
    plt.figure(figsize=(18, 10))
    for sv in sorted(satellite_errors.keys()):
        data = satellite_errors[sv]
        plt.plot(data['times'], data['errors'], 'o-', label=sv, markersize=4)
    plt.xlabel('Time (Broadcast Ephemeris toe)', fontsize=12)
    plt.ylabel('3D Error at toe (meters)', fontsize=12)
    plt.title('Broadcast vs SP3: Position Error at toe (Exact Match)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig('plots/all_satellites_error.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Statistics
    stats = {}
    for sv in satellite_errors:
        err_arr = np.array(satellite_errors[sv]['errors'])
        stats[sv] = {
            'mean_error_m': np.mean(err_arr),
            'std_error_m': np.std(err_arr),
            'rms_error_m': np.sqrt(np.mean(err_arr**2)),
            'max_error_m': np.max(err_arr),
            'min_error_m': np.min(err_arr),
            'count': len(err_arr)
        }

    stats_df = pd.DataFrame.from_dict(stats, orient='index')
    stats_df.index.name = 'Satellite'
    stats_df.to_csv("results/error_statistics.csv")

    # Overall summary
    all_errors = np.concatenate([np.array(v['errors']) for v in satellite_errors.values()])
    overall = {
        'Overall Mean Error (m)': np.mean(all_errors),
        'Overall RMS Error (m)': np.sqrt(np.mean(all_errors**2)),
        'Best Satellite': min(stats.items(), key=lambda x: x[1]['rms_error_m'])[0],
        'Worst Satellite': max(stats.items(), key=lambda x: x[1]['rms_error_m'])[0],
        'Total Matched Epochs': len(all_errors),
        'Number of Satellites': len(satellite_errors)
    }

    pd.DataFrame.from_dict(overall, orient='index', columns=['Value']).to_csv("results/overall_statistics.csv")

    print("\n=== Overall Statistics (at toe) ===")
    for k, v in overall.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")

    print("\n✅ All done! Check 'plots/', 'coordinates/', and 'results/' folders.")

if __name__ == "__main__":
    main()