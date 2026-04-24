from micpy import bin

conc_path = "/home/lokadm/Documents/presentations/repos/osw-chatbot/osw_files/downloads/OSW74d6882189514be48a0218c690305b51.conc1"  # adjust to your path

with bin.File(conc_path) as f:
    f.set_geometry(shape=(1500, 1, 500), spacing=(0.1, 0.1, 0.1))
    field = f.read_field(-1)

    fig, ax, cbar = bin.plot(field)
    fig.savefig("snippet_micpy_test.png", dpi=300)
