import avici
from avici import simulate_data, simulate_ldp_data
from avici.metrics import shd
from ldp.ldp_mechanism import GaussianLDP
import numpy as np


if __name__ == "__main__":

    g, x, interv = simulate_data(d=20, n=300, n_interv=0, path="ldp/config/ldp.yaml")
    # g, x, interv = simulate_data(d=20, n=300, n_interv=0, domain="lin-gauss")
    
    
    model = avici.load_pretrained(download="neurips-linear")
    ldp_model = avici.load_pretrained(checkpoint_dir="ldp/checkpoints/custom", expects_counts=False)

    # g: [d, d] predicted edge probabilities of the causal graph
    g_prob = model(x=x, interv=interv)
    g_pred = (g_prob > 0.5).astype(int)

    print(f"First 10 rows of x:\n{x[:10]}")

    print(f"Total edges: {np.sum(g)}")

    # print(f"g:\n{g}")
    #print(f"g_pred:\n{g_pred}")
    print(f"Normal Data Model SHD:\n{shd(g, g_pred)}")

    ldp_g_prob = ldp_model(x=x, interv=interv)
    ldp_g_pred = (ldp_g_prob > 0.5).astype(int)

    print(f"LDP Data Model SHD:\n{shd(g, ldp_g_pred)}")

