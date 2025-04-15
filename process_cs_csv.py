import os
import time
import logging
import argparse
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_cs_csv.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_BACKUP_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TEXT_COLUMNS = [
    "title", "abstract", "conclusion", "Chatgpt Response", "Key Takeaways", 
    "Importance", "Model/Method Proposed", "Performance", "Effectiveness", "Future Works"
]
DEFAULT_METADATA_COLUMNS = ["paperID", "venue", "year", "url", "authors"] # Example, adjust as needed
DEFAULT_CHUNK_SIZE = 1000 # Smaller chunk size suitable for abstracts/titles
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_PANDAS_CHUNKSIZE = 10000 # Number of rows to read from CSV at a time
DEFAULT_BATCH_SIZE = 500 # Number of documents to add to Chroma at once
DEFAULT_EMBEDDING_BATCH_SIZE = 64 # Batch size for the embedding model itself
DEFAULT_ID_COLUMN = "paperID"
ERROR_HANDLING_BATCH_SIZE = 50 # Process in smaller batches for error handling

class CsPaperCsvProcessor:
    def __init__(
        self,
        input_csv: str,
        output_dir: str,
        text_columns: List[str] = DEFAULT_TEXT_COLUMNS,
        metadata_columns: List[str] = DEFAULT_METADATA_COLUMNS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        pandas_chunksize: int = DEFAULT_PANDAS_CHUNKSIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        shard_id: int = 0,
        total_shards: int = 1,
        resume: bool = True,
        id_column: str = "paperID" # Assume 'paperID' is the unique identifier
    ):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.text_columns = text_columns
        self.metadata_columns = metadata_columns
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.pandas_chunksize = pandas_chunksize
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size # Pass this to HuggingFaceEmbeddings if applicable/needed
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.id_column = id_column
        self.resume = True  # Always try to resume, ignore the input parameter

        # --- Directory Setup ---
        shard_output_dir = output_dir
        if total_shards > 1:
            shard_output_dir = os.path.join(output_dir, f"shard_{shard_id}")
            logger.info(f"Sharding enabled: This is shard {shard_id}/{total_shards}, using output directory: {shard_output_dir}")
        os.makedirs(shard_output_dir, exist_ok=True)
        self.shard_output_dir = shard_output_dir

        # --- Checkpointing Setup ---
        self.checkpoint_file = os.path.join(shard_output_dir, "checkpoint.json")
        self.start_row = 0
        self.processed_rows_count = 0
        self.total_chunks_added = 0
        
        # Always try to load checkpoint if exists (auto-resume)
        self._load_checkpoint()
        # Never delete checkpoint even if empty - we'll write a new one as processing starts

        # --- Initialize Components ---
        logger.info("Initializing components...")
        self.device = self._get_best_available_device()
        logger.info(f"Using device: {self.device}")
        self.embeddings = self._init_embeddings()
        self.vectorstore = self._init_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""] # Standard separators
        )
        logger.info("Initialization complete.")


    def _get_best_available_device(self) -> str:
        """Get the best available device (CUDA GPU, MPS, or CPU)"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not found. Defaulting to CPU.")
            return "cpu"
        except Exception as e:
            logger.warning(f"Could not detect optimal device: {e}. Defaulting to CPU.")
            return "cpu"

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings with error handling."""
        logger.info(f"Initializing embedding model: {self.embedding_model} on device: {self.device}")
        model_kwargs = {'device': self.device}
        # encode_kwargs = {'batch_size': self.embedding_batch_size} # control embedding batch size

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs=model_kwargs,
                # encode_kwargs=encode_kwargs # May need adjustment based on langchain version
            )
            # Test embedding a short text
            embeddings.embed_query("test")
            logger.info(f"Successfully initialized primary embedding model: {self.embedding_model}")
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing primary embedding model '{self.embedding_model}': {e}")
            logger.warning(f"Falling back to backup embedding model: {DEFAULT_BACKUP_EMBEDDING_MODEL}")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name=DEFAULT_BACKUP_EMBEDDING_MODEL,
                    model_kwargs=model_kwargs,
                     # encode_kwargs=encode_kwargs
                )
                embeddings.embed_query("test")
                logger.info(f"Successfully initialized backup embedding model: {DEFAULT_BACKUP_EMBEDDING_MODEL}")
                self.embedding_model = DEFAULT_BACKUP_EMBEDDING_MODEL # Update the model name being used
                return embeddings
            except Exception as e_backup:
                logger.critical(f"Fatal Error: Could not initialize backup embedding model '{DEFAULT_BACKUP_EMBEDDING_MODEL}': {e_backup}")
                raise RuntimeError("Failed to initialize any embedding model.") from e_backup

    def _init_vectorstore(self) -> Chroma:
        """Initialize Chroma vector store."""
        logger.info(f"Initializing Chroma vector store at: {self.shard_output_dir}")
        try:
            vectorstore = Chroma(
                persist_directory=self.shard_output_dir,
                embedding_function=self.embeddings
            )
            # Test connection (optional, Chroma handles lazy init)
            # vectorstore.get()
            logger.info("Successfully initialized Chroma vector store.")
            return vectorstore
        except Exception as e:
            logger.critical(f"Fatal Error: Could not initialize Chroma vector store at '{self.shard_output_dir}': {e}")
            raise RuntimeError("Failed to initialize vector store.") from e

    def _load_checkpoint(self):
        """Load progress from the checkpoint file. Always try to resume if checkpoint exists."""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                self.start_row = checkpoint_data.get("processed_rows_count", 0)
                self.processed_rows_count = self.start_row # Start counting from the checkpoint
                self.total_chunks_added = checkpoint_data.get("total_chunks_added", 0)
                
                # Check if the checkpoint contains valid data to resume from
                if self.start_row > 0 or self.total_chunks_added > 0:
                    logger.info(f"Auto-resuming from checkpoint: Starting after row {self.start_row}. Previously added {self.total_chunks_added} chunks.")
                else:
                    logger.info("Found checkpoint file but with no progress recorded. Starting from beginning.")
            except json.JSONDecodeError:
                logger.error(f"Error reading checkpoint file {self.checkpoint_file}. Starting from scratch.")
                self.start_row = 0
                self.processed_rows_count = 0
                self.total_chunks_added = 0
            except Exception as e:
                logger.error(f"Unexpected error loading checkpoint: {e}. Starting from scratch.")
                self.start_row = 0
                self.processed_rows_count = 0
                self.total_chunks_added = 0
        else:
            logger.info("No checkpoint file found. Starting from the beginning.")

    def _save_checkpoint(self):
        """Save current progress to the checkpoint file."""
        checkpoint_data = {
            "processed_rows_count": self.processed_rows_count,
            "total_chunks_added": self.total_chunks_added,
            "timestamp": time.time()
        }
        try:
            # Write to a temporary file first, then rename for atomicity
            temp_file = self.checkpoint_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=4)
            os.replace(temp_file, self.checkpoint_file)
            logger.debug(f"Checkpoint saved: Processed up to row {self.processed_rows_count}, {self.total_chunks_added} total chunks added.")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")


    def _get_csv_reader(self) -> tuple[pd.io.parsers.readers.TextFileReader, int, Optional[int]]:
        """Creates a pandas CSV reader iterator with sharding support and returns necessary context."""
        logger.info(f"Setting up CSV reader for {self.input_csv} with chunksize {self.pandas_chunksize}")

        # Estimate total rows for sharding (optional but helpful for progress)
        total_rows_estimated = None
        # The row estimation logic is commented out, keeping it that way.
        # try:
        #     pass
        # except Exception as e:
        #     logger.warning(f"Could not quickly estimate total rows: {e}")


        # Calculate rows for this shard
        skiprows = 0
        nrows = None
        if self.total_shards > 1:
            if total_rows_estimated:
                rows_per_shard = total_rows_estimated // self.total_shards
                shard_start_row = self.shard_id * rows_per_shard
                shard_end_row = (self.shard_id + 1) * rows_per_shard if self.shard_id < self.total_shards - 1 else total_rows_estimated

                # Add 1 because skiprows is 0-based index *after* header
                skiprows = shard_start_row + 1 if shard_start_row > 0 else 0
                nrows = shard_end_row - shard_start_row
                logger.info(f"Shard {self.shard_id}: Reading approx {nrows} rows starting after row {skiprows} (1-based header).")
            else:
                 # If estimation failed, read the whole file and filter chunks - less efficient
                 logger.warning("Cannot estimate rows for sharding, will filter chunks based on index.")


        # Adjust skiprows based on checkpoint
        if self.start_row > 0:
             if self.total_shards > 1:
                 # We need to skip rows *within* the shard's range that were already processed
                 # Determine the global row index where this shard *would* start without a checkpoint
                 shard_global_start_row_index = 0
                 if total_rows_estimated:
                      rows_per_shard = total_rows_estimated // self.total_shards
                      shard_global_start_row_index = (self.shard_id * rows_per_shard) + 1 # +1 for 1-based index
                 else:
                      # If we can't estimate, we can't accurately calculate the shard start index for resuming.
                      # This part requires total_rows_estimated to work correctly with sharding + resuming.
                      # For now, log a warning. The filtering logic later will handle sharding, but resuming might be inaccurate.
                      logger.warning("Cannot accurately adjust skip rows for shard checkpoint without total row estimation.")
                      # Defaulting to basic skip based on self.start_row, might read wrong rows for shards.
                      shard_global_start_row_index = skiprows # Use the originally calculated skiprows if any

                 # The number of rows *within this shard* to skip because they are already processed
                 rows_to_skip_in_shard = max(0, self.start_row - shard_global_start_row_index)
                 actual_skiprows = shard_global_start_row_index + rows_to_skip_in_shard

                 if nrows is not None:
                    # Adjust the number of rows to read for this shard based on skipped rows
                    nrows = max(0, nrows - rows_to_skip_in_shard)

                 logger.info(f"Checkpoint active: Adjusted shard {self.shard_id} to skip {actual_skiprows} total rows (relative to file start), reading max {nrows} rows.")
                 skiprows = actual_skiprows
             else:
                 # Simple case: no sharding, just skip processed rows
                 skiprows = self.start_row + 1 # +1 for header
                 logger.info(f"Checkpoint active: Resuming CSV read, skipping first {self.start_row} data rows (global index {skiprows}).")
                 nrows = None # Read till end from the resume point


        try:
            reader = pd.read_csv(
                self.input_csv,
                chunksize=self.pandas_chunksize,
                on_bad_lines='warn', # Log bad lines instead of failing
                low_memory=True, # May help with mixed types, depends on data
                skiprows=range(1, skiprows) if skiprows > 0 else None, # Skip rows *after* the header
                nrows=nrows,
                # dtype=str # Force all columns to string initially? Might prevent type errors.
                encoding='utf-8',
                encoding_errors='replace'
            )
            # Return the reader and the calculated skiprows and total_rows_estimated
            return reader, skiprows, total_rows_estimated
        except FileNotFoundError:
            logger.critical(f"Fatal Error: Input CSV file not found at {self.input_csv}")
            raise
        except Exception as e:
            logger.critical(f"Fatal Error: Failed to create CSV reader for {self.input_csv}: {e}")
            raise

    def process_csv(self):
        """Reads the CSV file in chunks and processes each chunk."""
        start_time = time.time()
        # Get the reader and the context needed later
        reader, skiprows, total_rows_estimated = self._get_csv_reader()

        all_docs_batch: List[Document] = []
        last_checkpoint_time = time.time()
        chunk_index = -1 # Start at -1 so first chunk is 0
        # Load checkpoint data if resuming to calculate progress correctly later
        checkpoint_data = {}
        if self.resume and os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load checkpoint data for final stats: {e}")

        logger.info("Starting CSV processing...")

        try:
            for chunk_df in reader:
                chunk_index += 1
                # Calculate the starting row number for this chunk in the *original* file
                chunk_start_row_global = skiprows + chunk_index * self.pandas_chunksize

                # --- Sharding (Filter method if row estimation failed) ---
                # This filtering is needed *only* if total_rows_estimated is None AND total_shards > 1
                if self.total_shards > 1 and total_rows_estimated is None:
                    # Determine the effective 'global' index of the first row in this chunk
                    # Note: This assumes chunks align perfectly, which might not be true if rows were skipped due to errors previously.
                    # A more robust method might involve tracking row numbers directly if possible.
                    # For simplicity, we use the calculated chunk_start_row_global, assuming it's close enough.
                    # We check if the *chunk* index belongs to this shard.
                    if chunk_index % self.total_shards != self.shard_id:
                        logger.debug(f"Skipping pandas chunk {chunk_index} (belongs to shard {chunk_index % self.total_shards}) based on chunk index.")
                        continue # Skip this chunk if it's not for the current shard
                    logger.debug(f"Processing pandas chunk {chunk_index} (assigned to shard {self.shard_id}) based on chunk index.")

                logger.info(f"Processing pandas chunk {chunk_index} (starts ~row {chunk_start_row_global + 1} in original file)...") # +1 for 1-based index
                chunk_docs = self._process_chunk_df(chunk_df, chunk_start_row_global)

                if chunk_docs:
                    all_docs_batch.extend(chunk_docs)
                    logger.debug(f"Accumulated {len(all_docs_batch)} documents for Chroma batch.")

                    # Add to Chroma in batches
                    if len(all_docs_batch) >= self.batch_size:
                        self._add_documents_to_vectorstore(all_docs_batch)
                        all_docs_batch = [] # Clear batch

                # Update overall processed row count *after* successful processing of the chunk
                self.processed_rows_count += len(chunk_df)

                # --- Checkpointing ---
                current_time = time.time()
                if current_time - last_checkpoint_time >= 60: # Checkpoint every minute
                    logger.info("Periodic checkpoint...")
                    self._save_checkpoint()
                    self.vectorstore.persist() # Persist Chroma data too
                    last_checkpoint_time = current_time

            # Add any remaining documents
            if all_docs_batch:
                logger.info(f"Adding final batch of {len(all_docs_batch)} documents.")
                self._add_documents_to_vectorstore(all_docs_batch)

            # Final save and persist
            logger.info("CSV processing finished. Saving final checkpoint and persisting vector store.")
            self._save_checkpoint()
            self.vectorstore.persist()

            end_time = time.time()
            # Calculate processed rows in this specific run
            processed_in_this_run = self.processed_rows_count - self.start_row
            # Calculate chunks added in this specific run
            prev_chunks = checkpoint_data.get('total_chunks_added', 0) if self.resume else 0
            chunks_added_in_this_run = self.total_chunks_added - prev_chunks

            logger.info(f"Successfully processed CSV file.")
            logger.info(f"Total rows processed in this run: {processed_in_this_run}")
            logger.info(f"Total chunks added to vector store in this run: {chunks_added_in_this_run}")
            logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Error during CSV processing (chunk {chunk_index}): {e}", exc_info=True)
            logger.error("Attempting to save checkpoint before exiting due to error.")
            self._save_checkpoint()
            self.vectorstore.persist() # Attempt to save whatever was processed
            raise # Re-raise the exception


    def _process_chunk_df(self, chunk_df: pd.DataFrame, chunk_start_row_global: int) -> List[Document]:
        """Processes a pandas DataFrame chunk to create Langchain Documents, handling errors in sub-batches."""
        all_sub_batch_docs: List[Document] = [] # Accumulate docs from successful sub-batches
        rows_processed_in_chunk = 0
        rows_skipped_in_chunk = 0
        num_rows_in_df = len(chunk_df)

        logger.debug(f"Processing DataFrame chunk with {num_rows_in_df} rows, starting from global file row ~{chunk_start_row_global + 1}")

        for i in range(0, num_rows_in_df, ERROR_HANDLING_BATCH_SIZE):
            sub_batch_df = chunk_df.iloc[i:i + ERROR_HANDLING_BATCH_SIZE]
            sub_batch_start_row_global = chunk_start_row_global + i
            sub_batch_end_row_global = sub_batch_start_row_global + len(sub_batch_df)
            sub_batch_docs = []
            
            logger.debug(f"Processing sub-batch: original file rows {sub_batch_start_row_global + 1} to {sub_batch_end_row_global}")

            try:
                for index, row in sub_batch_df.iterrows():
                    # 'index' is the original index from the full CSV if not reset
                    # Calculate the global row number for *this specific row*
                    # We need the offset within the sub_batch relative to the chunk_df's original index
                    row_offset_in_chunk = chunk_df.index.get_loc(index)
                    row_num_global = chunk_start_row_global + row_offset_in_chunk + 1 # +1 for 1-based index

                    # Combine text from specified columns
                    text_parts = [str(row.get(col, '')) for col in self.text_columns]
                    combined_text = "\n".join(filter(None, text_parts)) # Join non-empty parts with newline

                    if not combined_text or combined_text.isspace():
                        logger.debug(f"Skipping row {row_num_global} due to empty combined text.")
                        # This skip doesn't count towards the sub-batch failure skip
                        continue

                    # Extract metadata
                    metadata = {"source_csv": self.input_csv, "row_number": row_num_global}
                    paper_id = None
                    for col in self.metadata_columns:
                        col_value = row.get(col)
                        # Convert non-serializable types like numpy int64 to standard Python types
                        if isinstance(col_value, np.generic):
                            metadata[col] = col_value.item()
                        elif pd.isna(col_value):
                            metadata[col] = None # Store NaN as None (will be filtered later)
                        else:
                            metadata[col] = col_value

                        if col == self.id_column:
                            paper_id = metadata[col] # Store the ID separately if needed

                    if paper_id is None:
                        logger.warning(f"Row {row_num_global} does not have a value for the id_column '{self.id_column}'. Using row number as fallback ID.")
                        metadata[self.id_column] = f"row_{row_num_global}" # Fallback ID
                        paper_id = metadata[self.id_column] # Update paper_id with fallback


                    # Split text into chunks
                    try:
                        text_chunks = self.text_splitter.split_text(combined_text)
                    except Exception as e:
                        logger.error(f"Error splitting text for row {row_num_global} (ID: {paper_id}): {e}. Skipping row.")
                        # This error skips only the single row, doesn't fail the sub-batch yet
                        continue

                    for chunk_index, chunk_text in enumerate(text_chunks):
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_index"] = chunk_index
                        chunk_metadata["chunk_id"] = f"{paper_id}_chunk_{chunk_index}" # Use potentially fallback paper_id
                        doc = Document(page_content=chunk_text, metadata=chunk_metadata)
                        sub_batch_docs.append(doc)
                
                # If loop completes without error for the sub-batch
                all_sub_batch_docs.extend(sub_batch_docs)
                rows_processed_in_chunk += len(sub_batch_df) # Count rows in successfully processed sub-batch
                logger.debug(f"Successfully processed sub-batch (rows {sub_batch_start_row_global + 1}-{sub_batch_end_row_global}), generated {len(sub_batch_docs)} documents.")

            except Exception as e:
                # Catch any unexpected error during sub-batch processing
                logger.error(f"Error processing sub-batch (original file rows ~{sub_batch_start_row_global + 1} to {sub_batch_end_row_global}): {e}. Skipping this batch of {len(sub_batch_df)} rows.", exc_info=True)
                rows_skipped_in_chunk += len(sub_batch_df)
                # Continue to the next sub-batch
                continue

        logger.info(f"Finished processing DataFrame chunk. Successfully processed {rows_processed_in_chunk} rows, skipped {rows_skipped_in_chunk} rows due to sub-batch errors. Generated {len(all_sub_batch_docs)} total documents for this chunk.")
        return all_sub_batch_docs

    def _add_documents_to_vectorstore(self, documents: List[Document]):
        """Adds a batch of documents to the Chroma vector store with retries."""
        if not documents:
            return

        logger.info(f"Adding batch of {len(documents)} documents to Chroma...")
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            try:
                # --- Filter complex/unsupported metadata types ---
                logger.debug("Filtering complex metadata before adding to Chroma...")
                filtered_documents = filter_complex_metadata(documents)
                # -------------------------------------------------

                # Use add_documents for simplicity unless specific ID handling is needed
                # Pass the filtered documents to Chroma
                added_ids = self.vectorstore.add_documents(filtered_documents)

                # If using add_texts with specific IDs (requires more handling):
                # texts = [doc.page_content for doc in filtered_documents]
                # metadatas = [doc.metadata for doc in filtered_documents]
                # added_ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=valid_ids)

                self.total_chunks_added += len(added_ids) # Count successfully added docs
                logger.info(f"Successfully added {len(added_ids)} documents to Chroma. Total chunks in store: ~{self.total_chunks_added}") # Note: Chroma's count might differ slightly
                return # Success
            except Exception as e:
                attempt += 1
                logger.error(f"Error adding documents to Chroma (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Failed to add batch to Chroma.")
                    # Decide how to handle this: skip batch, raise error, etc.
                    # For now, we'll log and continue, but data is lost.
                    # Consider saving failed batches to a file for later processing.
                    return # Move on without raising, but log the failure


def main():
    parser = argparse.ArgumentParser(description="Process research paper abstracts from a CSV file, generate embeddings, and store in Chroma.")
    parser.add_argument("input_csv", help="Path to the input CSV file.")
    parser.add_argument("output_dir", help="Directory to save the Chroma vector store and logs.")
    parser.add_argument("--text_cols", nargs='+', default=DEFAULT_TEXT_COLUMNS,
                        help=f"Column names containing the text to embed (default: {DEFAULT_TEXT_COLUMNS})")
    parser.add_argument("--meta_cols", nargs='+', default=DEFAULT_METADATA_COLUMNS,
                        help=f"Column names to include as metadata (default: {DEFAULT_METADATA_COLUMNS})")
    parser.add_argument("--id_col", default="paperID", help="Column name for the unique paper identifier (default: paperID)")
    parser.add_argument("--embed_model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Hugging Face model name for embeddings (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Target size for text chunks (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap between text chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--pandas_chunksize", type=int, default=DEFAULT_PANDAS_CHUNKSIZE,
                        help=f"Number of rows for pandas to read at a time (default: {DEFAULT_PANDAS_CHUNKSIZE})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of documents to batch add to Chroma (default: {DEFAULT_BATCH_SIZE})")
    # parser.add_argument("--embed_batch_size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help="Batch size for the embedding model itself") # Add if needed
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for parallel processing (default: 0)")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards for parallel processing (default: 1)")
    # Keep the resume argument for backward compatibility but make it no-op
    parser.add_argument("--resume", action='store_true', help="Resume processing from the last checkpoint (now always active by default).")
    parser.add_argument("--force_restart", action='store_true', help="Force restart from beginning, ignoring any existing checkpoint.")

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logger.critical(f"Input CSV file not found: {args.input_csv}")
        return

    logger.info("Starting CS Paper CSV Processing...")
    logger.info(f"Input CSV: {args.input_csv}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Text Columns: {args.text_cols}")
    logger.info(f"Metadata Columns: {args.meta_cols}")
    logger.info(f"ID Column: {args.id_col}")
    logger.info(f"Embedding Model: {args.embed_model}")
    logger.info(f"Chunk Size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    logger.info(f"Pandas Chunksize: {args.pandas_chunksize}")
    logger.info(f"Chroma Batch Size: {args.batch_size}")
    logger.info(f"Sharding: {args.shard_id}/{args.total_shards}")
    
    # If force_restart is specified, delete the checkpoint
    output_dir = args.output_dir
    if args.total_shards > 1:
        output_dir = os.path.join(output_dir, f"shard_{args.shard_id}")
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    
    if args.force_restart and os.path.exists(checkpoint_file):
        logger.warning(f"--force_restart specified. Deleting existing checkpoint and starting from beginning.")
        try:
            os.remove(checkpoint_file)
        except Exception as e:
            logger.error(f"Error deleting checkpoint file: {e}")


    try:
        processor = CsPaperCsvProcessor(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            text_columns=args.text_cols,
            metadata_columns=args.meta_cols,
            embedding_model=args.embed_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            pandas_chunksize=args.pandas_chunksize,
            batch_size=args.batch_size,
            # embedding_batch_size=args.embed_batch_size, # Pass if added
            shard_id=args.shard_id,
            total_shards=args.total_shards,
            # resume flag is ignored now, always attempting to resume
            id_column=args.id_col
        )
        processor.process_csv()
        logger.info("Processing completed successfully.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        logger.critical("Processing failed.")

if __name__ == "__main__":
    main() 