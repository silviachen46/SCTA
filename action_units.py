class ActionUnit:
    def __init__(self, name, instruction):
        self.name = name
        self.instruction = instruction

    def __str__(self):
        return f"Name: {self.name}\nInstruction: {self.instruction}"
    
class ActionUnitWithContext(ActionUnit):
    def __init__(self, name, instruction, action_unit):
        super().__init__(name, instruction) 
        self.action_unit = action_unit
        self.codes = {}
        self.errors = {}
        self.curr_result = None
        self.debug_num = 0
        self.run_num = 0

    def add_code(self, code):
        self.codes.append(code)

    def add_error(self, error):
        self.errors.append(error)

general_action_units = {
    "general_unit": """
      You are to conduct Single Cell analysis on the given data. 
      Before start, use import warnings
      warnings.filterwarnings("ignore", category=UserWarning) to filter the warning
      
      You should first quality control & filter the data,
      PCA and compute UMAP,
      annotate with given functions and marker set marker_genes from markers.py,
      plot your UMAP by cell_type, sample_group, and leiden cluster,
      plot cluster distribution by BL and plot by frequency with given function.
      do DEG and GSEA.

      and then extract the t cell from previous adata into adata_t. t cell is with "T_cells" in its adata.obs['cell_type'].
      redo log transform, PCA and compute UMAP,
      
      annotate with given functions and marker set marker_genes_t from markers.py,

      plot your UMAP by cell_type, sample_group, and leiden cluster,
      plot cluster distribution by BL and then plot by frequency with given function.

      All required path has been specified for you as default parameter, you do not need to modify them. You are allowed to tune other given params.
      You are given the following helper functions, and you should fully leverage them to conduct analysis.
      The helper function have default param values, but you should decide the specific param value to use which makes sense the most biologically.
      

      You can import them using "from utils_agent import load_and_annotate_h5_files, filter_cells, normalize_log_transform, filter_lowqc_cells, save_high_var_gene, pca_and_plot_umap, annoatate_by_markers"
      You will need marker dict for annotation. Use from markers import marker_genes, marker_genes_t for the function.
      All functions with the show_... parameters allows you to look into the numerial information of the graphics.
      Important: All return types are marked for you. If a return type is an adata, you should write as adata = func(adata). Otherwise you will miss important information.

      def load_and_annotate_h5_files(folder_path="/Users/silviachen/Documents/Software/SCAagent/h5_file") -> adata;

      def filter_cells(adata, min_cells=3, min_genes=200, max_genes=50000) -> None:
      def normalize_log_transform(adata, target_sum = 1e4) -> None:
      def filter_lowqc_cells(adata, pct_counts_mt_upbound = 10, n_genes_by_counts = None, pct_counts_mt_lowbound = None) -> adata:
      def save_high_var_gene(adata, n_top_genes = 4000) -> adata:
      
      # You should only use one option from the following options.
      [OPTION 1] leveraging biological knowledge deciding cluster params
      def pca_and_plot_umap(
            adata, 
            n_pcs=40, 
            n_neighbors=10, 
            resolution=0.5, 
        ) -> None:

      [OPTION 2] Using statistical knowledge assisting with determining the cluster params
      # this function leverages a pca_and_umap_single_run(adata, n_pcs, n_neighbors, resolution) on inside, but you don't need to use it.
      # if used ARI, you should use the final_adata as the result and do not attempt to run pca_and_plot_umap again(as this ari function already did for you). You should also try to refine your choice on resolution later.
      def ari_grid_search_clustering(
            adata,
            n_pcs_list=[10, 20, 30],
            k_list=[5, 10, 15],
            resolution=0.5,
            plot=True
        ) -> final_adata, ari_df, best_label:
        

      # this function contains multiple options for showing different helpful results on later reviewing, you should decide which one to show.
      def annotate_by_markers(
            adata, 
            marker_gene_dict, 
            show_sample_groupinfo=True,
            show_centroid=False,
            show_raw_num=True,
            show_nearest_clusters=True,
            top_k=3
        )-> None:

      def display_umap(adata, target = "cell_type") -> None: # cell_type, BLnumber for target
      def plot_cluster_distribution_by_BL(adata, target = "cell_type") -> None: # or use 'leiden' label for target
      def plot_cluster_frequencies(
    adata,
    cluster_key="leiden",
    group_key="sample_group",
    sample_key="BL_number"
) -> None:

   def get_deg(adata, top_n = 20, groupby = "condition", reference = "Control") -> None:
   def get_enrichment_analysis(adata, target = "CP", reference = "Control", groupby = "condition", file_path = "/Users/silviachen/Documents/Software/SCAagent/h5_file/h.all.v2024.1.Hs.symbols.gmt", top_n = 10) -> None:
      
      
"""
}