marker_genes = {
    "T_cells": [
        "CD3D", "CD3E",  # General T cell markers
        "CD4",  # Helper T cells
        "CD8A", "GZMB",  # Cytotoxic T cells
        "IL7R",  # Naïve/memory T cells
        "PRSS1"  # Specialized PRSS1-high T cells
    ],
    # "T_cells": [
    #     # General T cell markers
    #     "CD3D", "CD3E", "CD3G",  # TCR complex components
    #     "TRAC", "TRBC1", "TRBC2",  # TCR alpha and beta chains
    #     "PTPRC",  # CD45 (common leukocyte antigen)

    #     # Helper and cytotoxic T cell markers
    #     "CD4",  # Helper T cells
    #     "CD8A", "CD8B",  # Cytotoxic T cells

    #     # Activation and cytotoxicity markers
    #     "GZMA", "GZMB", "PRF1",  # Granzyme A/B, Perforin (cytotoxic activity)
    #     "NKG7", "GNLY",  # Granulysin (cytotoxic and NK-like functions)
    #     "IFNG", "TNF",  # Pro-inflammatory cytokines
    #     "LAG3", "PDCD1 (PD-1)", "CTLA4",  # Exhaustion and immune checkpoints
    #     "HLA-DRA", "HLA-DRB1",  # Activation markers (HLA-DR expression)

    #     # Memory and naïve markers
    #     "IL7R",  # IL-7 Receptor (Naïve and Memory T cells)
    #     "CCR7", "SELL",  # Naïve T cell markers
    #     "TCF7", "LEF1",  # Stem-like and memory T cells

    #     # Proliferation markers
    #     "MKI67", "TOP2A",  # Ki-67 and Topoisomerase II (cycling cells)

    #     # Specialized subsets
    #     "FOXP3", "IL2RA", "CTLA4",  # Regulatory T cells (Tregs)
    #     "TBX21", "RORC", "GATA3",  # Th1, Th17, and Th2 lineage factors
    #     "CCR6", "CXCR3", "CCR4"  # Chemokine receptors defining T cell subsets

    #     # Specialized PRSS1-high T cells (as in your original list)
    #     "PRSS1"
    # ],
    "B_cells": [
        "MS4A1", "CD79A", "IGKC",  # General B cell markers
        "SDC1", "JCHAIN", "IGHG1"  # Plasma cells (antibody-secreting)
    ],
    "Mast_cells": [
        "TPSAB1", "CPA3", "KIT"  # Classic mast cell markers
    ],
    "Myeloid_cells": [
        # Monocytes
        "CD14", "LYZ", "FCN1",
        # Macrophages
        "CD68", "FCGR3A",  # General
        "APOE", "C1QA", "C1QB",  # APOE-high macrophages
        "S100A9", "S100A8",  # Inflammatory macrophages
        "MKI67", "TOP2A",  # Proliferating macrophages
        # Dendritic Cells (DCs)
        "CD1C", "CLEC9A", "LAMP3"
    ],
    # "Other": [
    #     "PECAM1", "VWF", "CDH5",  # Endothelial cells
    #     "COL1A1", "COL1A2", "ACTA2",  # Pancreatic Stellate Cells (PSCs)
    #     "IRF7", "TCF4", "LILRA4",  # Plasmacytoid Dendritic Cells (pDCs)
    #     "KRT19", "KRT7", "EPCAM"  # Epithelial (ductal) cells
    # ]
}

marker_genes_t = {
    "CD4_T_cells_CCR6": ["CCR6", "IL17A", "RORC"],  # Th17 cells
    "CD8_Tc1": ["GZMA", "PRF1", "IFNG"],  # Cytotoxic CD8+ T cells (Tc1)
    "CD4_T_cells_FTH1": ["FTH1", "HSP90AA1", "SOD2"],  # FTH1-high CD4+ T cells
    "CD4_T_cells_BAG3": ["BAG3", "HSPA1A", "DNAJB1"],  # BAG3-high CD4+ T cells
    "CD4_T_cells_Th1": ["TBX21", "IFNG", "CXCR3"],  # Th1 cells
    "CD8_Tc2_HLA-DR_GZMK": ["HLA-DRA", "GZMK", "LAG3"],  # Exhausted CD8+ T cells
    "CD8_Tc3_BAG3": ["BAG3", "HSPA1B", "DDIT4"],  # BAG3-high CD8+ T cells
    "CD8_Tc4": ["GZMH", "CCL5", "CD8A"],  # CD8+ T cell subset 4
    "NK_cells": ["NCAM1", "KLRF1", "NKG7"],  # Natural Killer (NK) cells
    "MALAT1_cells": ["MALAT1", "NEAT1", "XIST"],  # MALAT1+ cells
    "Treg": ["FOXP3", "IL2RA", "CTLA4"],  # Regulatory T cells
    "Naïve_CD4_T_cells": ["SELL", "CCR7", "LEF1"],  # Naïve CD4+ T cells
    "CD8_Tc5_NK-like": ["EOMES", "NKG7", "GNLY"],  # NK-like CD8+ T cells
}