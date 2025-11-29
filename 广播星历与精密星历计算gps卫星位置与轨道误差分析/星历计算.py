import numpy as np
import pandas as pd
import georinex as gr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ==============================
# 常量定义
# ==============================
GM = 3.986005e14          # 地球引力常数 (m^3/s^2)
OMEGA_E = 7.2921151467e-5 # 地球自转角速度 (rad/s)

# ==============================
# 辅助函数：解 Kepler 方程
# ==============================
def solve_kepler(M, e, tol=1e-12, max_iter=10):
    E = M.copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if np.all(np.abs(dE) < tol):
            break
    return E

# ==============================
# 辅助函数：从广播星历计算卫星位置（输入 t_sow: GPS 周内秒）
# ==============================
def calc_position_from_ephemeris(t_sow, eph):
    """
    t_sow: GPS seconds of week (0 ~ 604800)
    eph: pandas.Series with RINEX ephemeris parameters
    Returns: [X, Y, Z] in meters (ECEF)
    """
    # Extract parameters
    toe = eph['Toe']      # already in seconds of week
    M0 = eph['M0']
    e = eph['Eccentricity']
    sqrtA = eph['sqrtA']
    omega = eph['omega']
    i0 = eph['Io']
    Omega0 = eph['Omega0']
    Delta_n = eph['DeltaN']
    Cuc = eph['Cuc']
    Cus = eph['Cus']
    Cic = eph['Cic']
    Cis = eph['Cis']
    Crc = eph['Crc']
    Crs = eph['Crs']
    idot = eph['IDOT']
    Omegadot = eph['OmegaDot']

    A = sqrtA ** 2
    n0 = np.sqrt(GM / (A ** 3))
    n = n0 + Delta_n

    # Time from ephemeris reference (tk)
    tk = t_sow - toe
    if tk > 302400:
        tk -= 604800
    elif tk < -302400:
        tk += 604800

    # Mean anomaly
    M = M0 + n * tk

    # Solve Kepler's equation
    E = solve_kepler(M, e)

    # True anomaly
    sinv = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
    cosv = (np.cos(E) - e) / (1 - e * np.cos(E))
    v = np.arctan2(sinv, cosv)

    # Argument of latitude
    u0 = v + omega

    # Radius
    r0 = A * (1 - e * np.cos(E))

    # Perturbations
    delta_u = Cuc * np.cos(2*u0) + Cus * np.sin(2*u0)
    delta_r = Crc * np.cos(2*u0) + Crs * np.sin(2*u0)
    delta_i = Cic * np.cos(2*u0) + Cis * np.sin(2*u0)

    u = u0 + delta_u
    r = r0 + delta_r
    i = i0 + delta_i + idot * tk

    # Orbital plane coordinates
    x_prime = r * np.cos(u)
    y_prime = r * np.sin(u)

    # Corrected longitude of ascending node
    Omega = Omega0 + (Omegadot - OMEGA_E) * tk - OMEGA_E * t_sow

    # ECEF
    X = x_prime * np.cos(Omega) - y_prime * np.cos(i) * np.sin(Omega)
    Y = x_prime * np.sin(Omega) + y_prime * np.cos(i) * np.cos(Omega)
    Z = y_prime * np.sin(i)

    return np.array([X, Y, Z])

# ==============================
# 解析 SP3 文件（返回 GPS SOW）
# ==============================
def parse_sp3(file_path, base_sow):
    """
    base_sow: GPS seconds of week at 00:00:00 of the day (e.g., 172800 for Tuesday)
    Returns: dict { 'G01': { 'times_sow': [...], 'pos': [...] } }
    """
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    current_sow = None
    for line in lines:
        if line.startswith('*'):
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            year, month, day = int(parts[1]), int(parts[2]), int(parts[3])
            hour, minute = int(parts[4]), int(parts[5])
            sec = float(parts[6])
            # Convert to seconds of day, then to SOW
            sod = hour * 3600 + minute * 60 + sec
            current_sow = base_sow + sod
        elif line.startswith('P') and len(line) > 50:
            # SP3 format: P<sys><PRN>
            sys_char = line[1]
            prn_str = line[2:4].strip()
            if sys_char == 'G' and prn_str.isdigit():
                sv_id = f"G{int(prn_str):02d}"  # e.g., G01
                try:
                    x = float(line[4:18]) * 1000   # km → m
                    y = float(line[18:32]) * 1000
                    z = float(line[32:46]) * 1000
                    if sv_id not in data:
                        data[sv_id] = {'times_sow': [], 'pos': []}
                    data[sv_id]['times_sow'].append(current_sow)
                    data[sv_id]['pos'].append([x, y, z])
                except Exception as e:
                    continue
    return data

# ==============================
# 保存坐标到文件
# ==============================
def save_coordinates_to_file(sv, times_sow, positions, filename):
    df = pd.DataFrame({
        'time_sow': times_sow,
        'X_m': positions[:, 0],
        'Y_m': positions[:, 1],
        'Z_m': positions[:, 2]
    })
    df.to_csv(filename, index=False)
    print(f"  Coordinates saved to {filename}")

# ==============================
# 主程序
# ==============================
def main():
    # 文件路径（请根据实际路径修改）
    brdc_file = "BRDC00GOP_R_20253220000_01D_MN.rnx"
    sp3_file = "WUM0MGXRAP_20253220000_01D_05M_ORB.SP3"

    # 创建输出目录
    os.makedirs("plots", exist_ok=True)
    os.makedirs("coordinates", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # === 关键：设置 GPS 周内秒基准 ===
    # 2025-11-18 is Tuesday.
    # GPS week starts on Sunday → Sunday=0, Monday=1, Tuesday=2
    # So 2025-11-18 00:00:00 = 2 * 86400 = 172800 seconds into GPS week
    BASE_SOW = 2 * 86400  # 172800

    # 加载广播星历
    print("Loading broadcast ephemeris...")
    nav = gr.load(brdc_file)
    gps_nav = nav.where(nav['sv'] != '', drop=True)
    gps_nav = gps_nav.sel(sv=[sv for sv in gps_nav.sv.values if sv.startswith('G')])

    # 展平为 DataFrame
    df_list = []
    for sv in gps_nav.sv.values:
        ds = gps_nav.sel(sv=sv).dropna(dim='time', how='all')
        for t in ds.time.values:
            rec = ds.sel(time=t).to_pandas()
            if not rec.isnull().all():
                rec['sv'] = sv
                rec['epoch'] = pd.Timestamp(t)
                # Convert RINEX time to GPS SOW
                dt = pd.Timestamp(t).to_pydatetime()
                # Assume dt is GPS time (no leap second correction needed for IGS products)
                sod = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
                # 2025-11-18 is day 2 of GPS week
                rec['t_sow'] = BASE_SOW + sod
                df_list.append(rec)
    eph_df = pd.concat(df_list, axis=1).T.reset_index(drop=True)

    satellites = sorted(eph_df['sv'].unique())
    print(f"Found GPS satellites: {satellites}")

    # 加载精密星历
    print("Loading precise ephemeris (SP3)...")
    sp3_data = parse_sp3(sp3_file, BASE_SOW)

    # 时间设置（in SOW）
    t_all = BASE_SOW + np.arange(0, 86401, 30)      # every 30 sec
    t_compare = BASE_SOW + np.arange(0, 86401, 900) # every 15 min

    all_errors_stats = {}
    all_errors_data = {}

    for sv in satellites:
        print(f"Processing {sv}...")

        # 获取该卫星所有星历
        sv_ephs = eph_df[eph_df['sv'] == sv].copy()
        if sv_ephs.empty:
            print(f"  No ephemeris for {sv}")
            continue

        # 计算广播轨道（动态选择星历）
        brdc_pos = []
        for t in t_all:
            # Find ephemeris with smallest |t - Toe| and within ±4 hours
            sv_ephs['dt'] = np.abs(sv_ephs['Toe'] - t)
            valid = sv_ephs[sv_ephs['dt'] <= 7200]  # 4 hours = 14400 sec
            if valid.empty:
                valid = sv_ephs  # fallback
            best_eph = valid.loc[valid['dt'].idxmin()]
            pos = calc_position_from_ephemeris(t, best_eph)
            brdc_pos.append(pos)
        brdc_pos = np.array(brdc_pos)

        # Save coordinates
        coord_filename = f"coordinates/{sv}_broadcast_coordinates.csv"
        save_coordinates_to_file(sv, t_all, brdc_pos, coord_filename)

        # Plot orbit
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(brdc_pos[:, 0]/1000, brdc_pos[:, 1]/1000, brdc_pos[:, 2]/1000, linewidth=0.8)
        ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
        ax.set_title(f'{sv} Broadcast Orbit')
        plt.savefig(f"plots/{sv}_orbit.png", dpi=150)
        plt.close()

        # Compare with SP3
        if sv not in sp3_data:
            print(f"  Warning: {sv} not found in SP3 file.")
            continue

        sp3_times = np.array(sp3_data[sv]['times_sow'])
        sp3_pos = np.array(sp3_data[sv]['pos'])

        errors = []
        valid_times = []
        for t in t_compare:
            idx = np.argmin(np.abs(sp3_times - t))
            if abs(sp3_times[idx] - t) > 300:  # >5 min
                continue
            true_pos = sp3_pos[idx]
            # Use dynamic ephemeris selection again
            sv_ephs['dt'] = np.abs(sv_ephs['Toe'] - t)
            valid_eph = sv_ephs[sv_ephs['dt'] <= 7200]
            if valid_eph.empty:
                valid_eph = sv_ephs
            best_eph = valid_eph.loc[valid_eph['dt'].idxmin()]
            brdc_at_t = calc_position_from_ephemeris(t, best_eph)
            err = np.linalg.norm(brdc_at_t - true_pos)
            errors.append(err)
            valid_times.append(t)

        errors = np.array(errors)
        valid_times = np.array(valid_times)

        all_errors_data[sv] = {'times_sow': valid_times, 'errors': errors}

        if len(errors) > 0:
            stats = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'max': np.max(errors),
                'min': np.min(errors),
                'rms': np.sqrt(np.mean(errors**2))
            }
            all_errors_stats[sv] = stats

            # Plot error
            plt.figure(figsize=(10, 4))
            plt.plot((valid_times - BASE_SOW)/3600, errors, 'o-', markersize=4)
            plt.xlabel('Time (hours since 00:00)')
            plt.ylabel('Orbit Error (meters)')
            plt.title(f'{sv} Broadcast vs Precise Orbit Error')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"plots/{sv}_error.png", dpi=150)
            plt.close()
        else:
            all_errors_stats[sv] = None

    # ==============================
    # 绘制汇总图
    # ==============================
    if all_errors_data:
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(satellites)))
        for i, sv in enumerate(satellites):
            if sv in all_errors_data and len(all_errors_data[sv]['errors']) > 0:
                data = all_errors_data[sv]
                plt.plot((data['times_sow'] - BASE_SOW)/3600, data['errors'],
                         color=colors[i], label=sv, linewidth=1, alpha=0.8)
        plt.xlabel('Time (hours since 00:00)')
        plt.ylabel('Orbit Error (meters)')
        plt.title('Broadcast vs Precise Orbit Errors for All GPS Satellites')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("plots/all_satellites_errors_timeseries.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 统计柱状图
    valid_sats = [sv for sv in satellites if all_errors_stats.get(sv) is not None]
    if valid_sats:
        means = [all_errors_stats[sv]['mean'] for sv in valid_sats]
        stds = [all_errors_stats[sv]['std'] for sv in valid_sats]
        rmss = [all_errors_stats[sv]['rms'] for sv in valid_sats]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_pos = np.arange(len(valid_sats))
        ax1.bar(x_pos - 0.2, means, 0.4, yerr=stds, capsize=5, label='Mean Error')
        ax1.bar(x_pos + 0.2, rmss, 0.4, label='RMS Error', alpha=0.7)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(valid_sats, rotation=45)
        ax1.set_ylabel('Error (m)')
        ax1.set_title('Mean and RMS Orbit Errors')
        ax1.legend()
        ax1.grid(alpha=0.3)

        max_errors = [all_errors_stats[sv]['max'] for sv in valid_sats]
        ax2.bar(valid_sats, max_errors, color='lightcoral', alpha=0.7)
        ax2.set_ylabel('Max Error (m)')
        ax2.set_title('Maximum Orbit Errors')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("plots/all_satellites_errors_stats.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 保存结果
    if all_errors_stats:
        stats_df = pd.DataFrame.from_dict(all_errors_stats, orient='index')
        stats_df.index.name = 'Satellite'
        stats_df.to_csv("results/orbit_errors_statistics.csv")
        print("Error statistics saved to results/orbit_errors_statistics.csv")

        valid_stats = {k: v for k, v in all_errors_stats.items() if v is not None}
        if valid_stats:
            all_means = [s['mean'] for s in valid_stats.values()]
            all_rmss = [s['rms'] for s in valid_stats.values()]
            overall = {
                'Overall Mean Error (m)': np.mean(all_means),
                'Overall RMS Error (m)': np.mean(all_rmss),
                'Best Satellite': min(valid_stats.items(), key=lambda x: x[1]['rms'])[0],
                'Worst Satellite': max(valid_stats.items(), key=lambda x: x[1]['rms'])[0],
                'Number of Satellites': len(valid_stats)
            }
            pd.DataFrame.from_dict(overall, orient='index', columns=['Value']).to_csv("results/overall_statistics.csv")
            print("\n=== Overall Statistics ===")
            for k, v in overall.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.2f}")
                else:
                    print(f"{k}: {v}")

    print("\nAll done!")
    print("Plots saved in 'plots/' folder")
    print("Coordinates saved in 'coordinates/' folder")
    print("Results saved in 'results/' folder")


if __name__ == "__main__":
    main()