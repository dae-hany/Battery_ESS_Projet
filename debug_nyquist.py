import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_nyquist_b0018():
    path = 'NASA_PCOE_DATA/B0018.mat'
    if not os.path.exists(path):
        print("File not found")
        return

    mat = scipy.io.loadmat(path)
    data = mat['B0018'][0, 0]['cycle'][0]
    
    impedance_cycles = []
    for i, c in enumerate(data):
        if c['type'][0] == 'impedance':
            if 'data' in c.dtype.names and c['data'].size > 0:
                impedance_cycles.append(c['data'][0, 0])
    
    print(f"Found {len(impedance_cycles)} impedance cycles.")
    
    if not impedance_cycles:
        return

    # Select First, Middle, Last
    indices = [0, len(impedance_cycles)//2, len(impedance_cycles)-1]
    
    plt.figure(figsize=(10, 8))
    
    for idx in indices:
        cycle = impedance_cycles[idx]
        
        re = None
        im = None
        
        if 're' in cycle.dtype.names:
            re = cycle['re'][0].flatten()
            im = cycle['im'][0].flatten()
        elif 'Rectified_Impedance' in cycle.dtype.names:
             Z = cycle['Rectified_Impedance'].flatten()
             re = np.real(Z)
             im = np.imag(Z)
             
        if re is not None:
            plt.plot(re, -im, 'o-', label=f'Cycle Index {idx}')
            print(f"Cycle {idx}: Re range [{re.min():.4f}, {re.max():.4f}], Im range [{im.min():.4f}, {im.max():.4f}]")
            
    plt.xlabel('Re(Z)')
    plt.ylabel('-Im(Z)')
    plt.title('Nyquist Plots for B0018')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('debug_nyquist_b0018.png')
    print("Saved debug_nyquist_b0018.png")

if __name__ == "__main__":
    plot_nyquist_b0018()
