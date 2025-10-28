import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

N = 51            # Chiều dài bộ lọc (lẻ để có tâm đối xứng nguyên)
fc = 0.2          # Tần số cắt mong muốn (chuẩn hóa theo fs/2, ví dụ 0.2 * fs/2)
fs = 1.0          # Tần số lấy mẫu (chuẩn hóa là 1.0)
wc = 2 * np.pi * fc / fs # Tần số cắt chuẩn hóa (rad/sample) -> wc = 0.4*pi

# Đáp ứng xung lý tưởng (đã dịch chuyển để có pha tuyến tính)
alpha = (N - 1) / 2
n = np.arange(N)

# Sử dụng np.sinc, lưu ý định nghĩa sinc(x) = sin(pi*x)/(pi*x)
h_ideal = wc / np.pi * np.sinc(wc * (n - alpha) / np.pi)

# Tạo các hàm cửa sổ
windows = {
    'Chữ nhật': signal.windows.boxcar(N),
    'Bartlett': signal.windows.bartlett(N),
    'Hanning': signal.windows.hann(N),
    'Hamming': signal.windows.hamming(N),
    'Blackman': signal.windows.blackman(N)
}

# Áp dụng cửa sổ để tạo hệ số bộ lọc FIR
filters = {}
print("Hệ số bộ lọc h[n]:")
for name, win in windows.items():
    filters[name] = h_ideal * win
    print(f"\n--- {name} ---")
    print(f"h[0] = {filters[name][0]:.6f}")
    print(f"h[{int(alpha)}] = {filters[name][int(alpha)]:.6f} (tâm)")
    print(f"h[{N-1}] = {filters[name][N-1]:.6f}")


# Tính toán và Vẽ Đáp ứng Tần số
num_freq_points = 8192
plt.style.use('seaborn-v0_8-whitegrid')

# Tạo figure và các subplot
fig1 = plt.figure(figsize=(12, 16)) # Kích thước lớn hơn để ảnh rõ nét khi lưu
gs = fig1.add_gridspec(2, 1) # GridSpec cho 2 hàng, 1 cột

# Subplot 1: Đáp ứng Biên độ (dB)
ax1 = fig1.add_subplot(gs[0, 0])
max_stopband_attenuation = {}
transition_widths = {}

for name, h in filters.items():
    w_rad, H = signal.freqz(h, worN=num_freq_points, fs=2 * np.pi)
    magnitude_db = 20 * np.log10(np.abs(H) + 1e-9)
    ax1.plot(w_rad / np.pi, magnitude_db, label=name, linewidth=1.5)

    #  Ước lượng các thông số
    # Độ suy giảm dải chắn (tìm giá trị dB lớn nhất trong dải chắn)
    stopband_start_freq_rad = wc + 0.1 * np.pi
    stopband_indices = np.where(w_rad >= stopband_start_freq_rad)[0]
    if len(stopband_indices) > 0:
        # Get the maximum value (least attenuation) in the stopband
        max_stopband_attenuation[name] = np.max(magnitude_db[stopband_indices])
    else:
        max_stopband_attenuation[name] = -np.inf # Could not determine

    # Độ rộng dải chuyển tiếp (ví dụ: từ -3dB đến -40dB)
    try:
        passband_end_idx = np.where(magnitude_db <= -3.0)[0][0]
        target_attenuation = -40.0 # dB
        stopband_start_idx = np.where(magnitude_db[passband_end_idx:] <= target_attenuation)[0][0] + passband_end_idx
        transition_widths[name] = (w_rad[stopband_start_idx] - w_rad[passband_end_idx]) / np.pi
    except IndexError:
        transition_widths[name] = np.nan

ax1.set_title(f'Đáp ứng Biên độ Bộ lọc FIR Thông thấp (N = {N})', fontsize=16)
ax1.set_xlabel(r'Tần số Chuẩn hóa ($\omega / \pi$)', fontsize=12)
ax1.set_ylabel('Biên độ (dB)', fontsize=12)
ax1.set_ylim(-120, 5) # Giới hạn trục y để thấy rõ dải chắn
ax1.set_xlim(0, 1) # Giới hạn trục x từ 0 đến 1 (tức 0 đến pi)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
# Vẽ đường tần số cắt
ax1.axvline(wc / np.pi, color='k', linestyle=':', linewidth=1.5, label=fr'$\omega_c={wc/np.pi:.2f}\pi$')
# Vẽ các đường tham chiếu dB
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax1.axhline(-20, color='lightgray', linestyle=':', linewidth=0.5)
ax1.axhline(-40, color='lightgray', linestyle=':', linewidth=0.5)
ax1.axhline(-60, color='lightgray', linestyle=':', linewidth=0.5)
ax1.legend(fontsize=11)
ax1.tick_params(axis='both', which='major', labelsize=10)

#Subplot 2: Đáp ứng Pha
ax2 = fig1.add_subplot(gs[1, 0])
for name, h in filters.items():
    w_rad, H = signal.freqz(h, worN=num_freq_points, fs=2 * np.pi)
    phase = np.unwrap(np.angle(H)) # Mở pha để tránh bước nhảy 2pi
    # Hiệu chỉnh độ trễ nhóm tuyến tính alpha * w
    phase_adjusted = phase + alpha * w_rad
    ax2.plot(w_rad / np.pi, phase_adjusted, label=name, linewidth=1.5)

ax2.set_title('Đáp ứng Pha (đã hiệu chỉnh độ trễ nhóm) - FIR Thông thấp (N = {N})', fontsize=16)
ax2.set_xlabel(r'Tần số Chuẩn hóa ($\omega / \pi$)', fontsize=12)
ax2.set_ylabel('Pha (rad)', fontsize=12)
ax2.set_ylim(-0.2, 0.2) # Giới hạn trục y nhỏ để thấy pha gần 0
ax2.set_xlim(0, 1) # Giới hạn trục x từ 0 đến 1 (tức 0 đến pi)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.legend(fontsize=11)
ax2.tick_params(axis='both', which='major', labelsize=10)

fig1.tight_layout(pad=3.0) # Điều chỉnh khoảng cách giữa các subplot và lề
fig1.savefig('dapungbiendovadapungpha.png', dpi=300) # Lưu cả hai đồ thị vào một file
# plt.show() # Uncomment để hiển thị đồ thị

# In Bảng So sánh
print("\n--- So sánh Hiệu năng (Ước lượng) ---")
print(f"{'Cửa sổ':<15} | {'Suy giảm dải chắn (dB)':<25} | {'Dải chuyển tiếp (x pi)':<25}")
print("-" * 65)
for name in sorted(filters.keys()): # Sắp xếp tên cửa sổ
    atten_str = f"{max_stopband_attenuation[name]:.2f}" if max_stopband_attenuation[name] > -np.inf else "N/A"
    width_str = f"{transition_widths[name]:.3f}" if not np.isnan(transition_widths[name]) else "N/A (<-40dB?)"
    print(f"{name:<15} | {atten_str:<25} | {width_str:<25}")

# (Tùy chọn) Vẽ các Hệ số Bộ lọc
fig2 = plt.figure(figsize=(12, 7)) # Kích thước cho đồ thị hệ số
ax3 = fig2.add_subplot(1, 1, 1)
marker_list = ['o', 's', '^', 'd', 'x']
marker_idx = 0
for name, h in filters.items():
    ax3.plot(n, h, marker=marker_list[marker_idx % len(marker_list)], linestyle='-', label=name, alpha=0.8, markersize=4, linewidth=1.0)
    marker_idx += 1
ax3.set_title(f'Hệ số Bộ lọc FIR (N = {N})', fontsize=16)
ax3.set_xlabel('Chỉ số mẫu (n)', fontsize=12)
ax3.set_ylabel('Giá trị h[n]', fontsize=12)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
ax3.legend(fontsize=11)
ax3.tick_params(axis='both', which='major', labelsize=10)
fig2.tight_layout(pad=3.0)
fig2.savefig('hesoboloc.png', dpi=300) # Lưu đồ thị hệ số vào file riêng