import matplotlib.pyplot as plt
import numpy as np

def fig1():
    beta = ['16','32','64','128','256']
    values = np.array([5.2881e+11,2.70012e+11,1.36453e+11,6.8594e+10,3.43896e+10])
    fig, ax = plt.subplots()

    ax.plot(beta, values/(2**30), marker='o')
    # ax.set_title('Thử nghiệm chút ...')
    ax.set_ylabel('Throughput (Gbps)')
    ax.set_xlabel('Beta')
    ax.grid()

    plt.show()

def fig2():
    Tp = [10,20,30,40,50]
    values = np.array([8.25585e+11,4.26606e+11,2.87612e+11,2.16933e+11,1.74139e+11])
    fig, ax = plt.subplots()

    ax.plot(Tp, values/(2**30), marker='o')
    ax.set_ylabel('Throughput (Gbps)')
    ax.set_xlabel('Pulse duration $T_p$ (femtosecond)')
    ax.grid()

    plt.show()

def fig3():
    cases = ["3","4","5","6","7"]

    thrs_on = []
    for c in cases:
        f = open(f"rts_on_{c}.data", "r")
        throughput = 0
        count = 0
        for x in f:
            throughput += float(x.split(" ")[7])
            count += 1
        thrs_on.append(throughput/count)

    thrs_off = []
    for c in cases:
        f = open(f"rts_off_{c}.data", "r")
        throughput = 0
        count = 0
        for x in f:
            throughput += float(x.split(" ")[7])
            count += 1
        thrs_off.append(throughput/count)
    
    thrs_on = np.array(thrs_on)
    thrs_off =  np.array(thrs_off)
    
    fig, ax = plt.subplots()
    ind = np.arange(len(cases))
    width = 0.2
    ax.bar(ind - width/2, thrs_on/(2**30), width, label='RTS/CTS on')
    ax.bar(ind + width/2, thrs_off/(2**30), width, label='RTS/CTS off')
    ax.legend()
    ax.grid()
    ax.autoscale_view()
    ax.set_ylabel('Throughput (Gbps)')
    ax.set_xlabel('Số lượng thiết bị')
    plt.show()

if __name__ == "__main__":
    fig3()