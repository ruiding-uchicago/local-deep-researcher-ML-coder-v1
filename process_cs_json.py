import os
import time
import logging
import argparse
import json
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata  # Added import for metadata filtering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process_cs_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
DEFAULT_BACKUP_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TEXT_KEYS = ["title", "abstract"]
DEFAULT_METADATA_KEYS = ["id", "submitter", "authors", "comments", "journal-ref", "doi", "report-no", "categories", "versions", "update_date", "authors_parsed"] # Example, adjust as needed
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 500 # Number of documents to add to Chroma at once
DEFAULT_EMBEDDING_BATCH_SIZE = 64
DEFAULT_ID_KEY = "id" # Key for the unique identifier in the JSON

class CsPaperJsonProcessor:
    def __init__(
        self,
        input_jsonl: str,
        output_dir: str,
        text_keys: List[str] = DEFAULT_TEXT_KEYS,
        metadata_keys: List[str] = DEFAULT_METADATA_KEYS,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = DEFAULT_BATCH_SIZE,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        shard_id: int = 0,
        total_shards: int = 1,
        resume: bool = True,  # Changed default to True
        id_key: str = DEFAULT_ID_KEY,
        max_lines: Optional[int] = None # Optional: Limit number of lines to process
    ):
        self.input_jsonl = input_jsonl
        self.output_dir = output_dir
        self.text_keys = text_keys
        self.metadata_keys = metadata_keys
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.shard_id = shard_id
        self.total_shards = total_shards
        self.id_key = id_key
        self.max_lines = max_lines
        self.resume = True  # Always try to resume, ignore the input parameter

        # --- Directory Setup ---
        shard_output_dir = output_dir
        if total_shards > 1:
            shard_output_dir = os.path.join(output_dir, f"shard_{shard_id}")
            logger.info(f"Sharding enabled: This is shard {shard_id}/{total_shards}, using output directory: {shard_output_dir}")
        os.makedirs(shard_output_dir, exist_ok=True)
        self.shard_output_dir = shard_output_dir

        # --- Checkpointing Setup ---
        self.checkpoint_file = os.path.join(shard_output_dir, "checkpoint_json.json")
        self.start_line = 0 # 0-based line index to start reading from
        self.processed_lines_count = 0 # Count of lines successfully processed *in this run*
        self.total_lines_processed_cumulative = 0 # Cumulative across runs (loaded from checkpoint)
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
        # Identical to the CSV version
        try:
            import torch
            if torch.cuda.is_available(): return "cuda"
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
            return "cpu"
        except ImportError: return "cpu"
        except Exception: return "cpu"

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize HuggingFace embeddings with error handling."""
        # Mostly identical to the CSV version
        logger.info(f"Initializing embedding model: {self.embedding_model} on device: {self.device}")
        model_kwargs = {'device': self.device}
        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model, model_kwargs=model_kwargs)
            embeddings.embed_query("test")
            logger.info(f"Successfully initialized primary embedding model: {self.embedding_model}")
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing primary embedding model '{self.embedding_model}': {e}")
            logger.warning(f"Falling back to backup embedding model: {DEFAULT_BACKUP_EMBEDDING_MODEL}")
            try:
                embeddings = HuggingFaceEmbeddings(model_name=DEFAULT_BACKUP_EMBEDDING_MODEL, model_kwargs=model_kwargs)
                embeddings.embed_query("test")
                logger.info(f"Successfully initialized backup embedding model: {DEFAULT_BACKUP_EMBEDDING_MODEL}")
                self.embedding_model = DEFAULT_BACKUP_EMBEDDING_MODEL
                return embeddings
            except Exception as e_backup:
                logger.critical(f"Fatal Error: Could not initialize backup embedding model '{DEFAULT_BACKUP_EMBEDDING_MODEL}': {e_backup}")
                raise RuntimeError("Failed to initialize any embedding model.") from e_backup

    def _init_vectorstore(self) -> Chroma:
        """Initialize Chroma vector store."""
        # Identical to the CSV version
        logger.info(f"Initializing Chroma vector store at: {self.shard_output_dir}")
        try:
            vectorstore = Chroma(persist_directory=self.shard_output_dir, embedding_function=self.embeddings)
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
                # start_line is the index of the *next* line to process
                self.start_line = checkpoint_data.get("next_line_index", 0)
                self.total_lines_processed_cumulative = checkpoint_data.get("total_lines_processed_cumulative", 0)
                self.total_chunks_added = checkpoint_data.get("total_chunks_added", 0)
                
                # Check if the checkpoint contains valid data to resume from
                if self.start_line > 0 or self.total_chunks_added > 0 or self.total_lines_processed_cumulative > 0:
                    logger.info(f"Auto-resuming from checkpoint: Starting at line index {self.start_line}. Previously processed {self.total_lines_processed_cumulative} lines cumulatively, added {self.total_chunks_added} chunks.")
                else:
                    logger.info("Found checkpoint file but with no progress recorded. Starting from beginning.")
            except json.JSONDecodeError:
                logger.error(f"Error reading checkpoint file {self.checkpoint_file}. Starting from scratch.")
                self._reset_progress()
            except Exception as e:
                logger.error(f"Unexpected error loading checkpoint: {e}. Starting from scratch.")
                self._reset_progress()
        else:
            logger.info("No checkpoint file found. Starting from the beginning.")
            self._reset_progress()

    def _reset_progress(self):
        """Resets progress counters."""
        self.start_line = 0
        self.processed_lines_count = 0
        self.total_lines_processed_cumulative = 0
        self.total_chunks_added = 0

    def _save_checkpoint(self, current_line_index: int):
        """Save current progress to the checkpoint file."""
        # The next line to process is the one *after* the current one
        next_line_index = current_line_index + 1
        # Update cumulative count with lines processed *in this specific run*
        current_cumulative = self.total_lines_processed_cumulative + self.processed_lines_count

        checkpoint_data = {
            "next_line_index": next_line_index, # Save the index of the next line to process
            "total_lines_processed_cumulative": current_cumulative, # Update cumulative total
            "total_chunks_added": self.total_chunks_added,
            "timestamp": time.time()
        }
        try:
            temp_file = self.checkpoint_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=4)
            os.replace(temp_file, self.checkpoint_file)
            logger.debug(f"Checkpoint saved: Next line to process is {next_line_index}. Total lines processed cumulatively: {current_cumulative}. Total chunks added: {self.total_chunks_added}.")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def _parse_json_line(self, line: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Safely parses a single line of JSON."""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"Skipping line {line_num + 1}: Invalid JSON: '{line[:100]}...'") # Log line number (1-based)
            return None
        except Exception as e:
             logger.error(f"Skipping line {line_num + 1}: Unexpected error parsing JSON: {e}")
             return None

    def _process_single_json_object(self, data: Dict[str, Any], line_num: int) -> Optional[List[Document]]:
        """Processes a single JSON object to create Langchain Documents."""
        # Combine text from specified keys
        text_parts = [str(data.get(key, '')) for key in self.text_keys]
        combined_text = "\n".join(filter(None, text_parts))

        if not combined_text or combined_text.isspace():
            logger.debug(f"Skipping line {line_num + 1} due to empty combined text.")
            return None

        # Extract metadata
        metadata = {"source_jsonl": self.input_jsonl, "line_number": line_num + 1} # 1-based line number
        obj_id = None
        for key in self.metadata_keys:
            if key in data:
                 value = data[key]
                 # Simple check for basic types, convert others to string
                 if isinstance(value, (str, int, float, bool, list, dict)):
                      # Attempt to handle potential numpy types within lists/dicts if needed later
                      metadata[key] = value
                 else:
                     metadata[key] = str(value) # Convert complex types to string

            if key == self.id_key:
                 obj_id = metadata.get(key) # Store the ID

        if obj_id is None:
             logger.warning(f"Line {line_num + 1} does not have a value for the id_key '{self.id_key}'. Using line number as fallback ID.")
             obj_id = f"line_{line_num + 1}"
             metadata[self.id_key] = obj_id

        # Split text into chunks
        try:
            text_chunks = self.text_splitter.split_text(combined_text)
        except Exception as e:
             logger.error(f"Error splitting text for line {line_num + 1} (ID: {obj_id}): {e}. Skipping object.")
             return None

        documents = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_id"] = f"{obj_id}_chunk_{i}"
            doc = Document(page_content=chunk_text, metadata=chunk_metadata)
            documents.append(doc)

        return documents

    def process_jsonl(self):
        """Reads the JSON Lines file line by line and processes each JSON object."""
        start_time = time.time()
        all_docs_batch: List[Document] = []
        last_checkpoint_time = time.time()
        lines_read_in_session = 0 # Count lines read from the file in this session
        error_count = 0

        logger.info(f"Starting JSONL processing from line index {self.start_line}...")

        try:
            with open(self.input_jsonl, 'r', encoding='utf-8', errors='replace') as f:
                # Use tqdm for progress bar, starting from the checkpointed line number
                # Note: Getting total lines can be slow for huge files.
                # total_lines = sum(1 for _ in open(self.input_jsonl, 'r', encoding='utf-8', errors='replace'))
                # logger.info(f"Estimated total lines: {total_lines}")
                # progress_bar = tqdm(f, total=total_lines, desc="Processing JSONL", initial=self.start_line, unit="lines")

                # Simpler approach: iterate line by line
                current_line_index = -1
                for line in f:
                    current_line_index += 1
                    lines_read_in_session += 1

                    # --- Line Skipping (Checkpoints) ---
                    if current_line_index < self.start_line:
                        if current_line_index == 0 and self.start_line > 0:
                             logger.info(f"Skipping to resume point (line index {self.start_line})...")
                        # TQDM update might be useful here if skipping many lines
                        continue # Skip lines already processed

                    # --- Max Lines Limit ---
                    if self.max_lines is not None and (self.total_lines_processed_cumulative + self.processed_lines_count) >= self.max_lines:
                        logger.info(f"Reached max_lines limit ({self.max_lines}). Stopping processing.")
                        break

                    # --- Sharding ---
                    if self.total_shards > 1 and current_line_index % self.total_shards != self.shard_id:
                        continue # Skip line if it belongs to another shard

                    # --- Process Line ---
                    json_data = self._parse_json_line(line, current_line_index)
                    if json_data is None:
                        error_count += 1
                        continue

                    documents = self._process_single_json_object(json_data, current_line_index)
                    if documents:
                        all_docs_batch.extend(documents)
                        self.processed_lines_count += 1 # Increment *only* if line was successfully processed into docs

                        # Add to Chroma in batches
                        if len(all_docs_batch) >= self.batch_size:
                            self._add_documents_to_vectorstore(all_docs_batch)
                            all_docs_batch = [] # Clear batch
                    else:
                        # Line parsed but resulted in no documents (e.g., empty text, splitting error)
                        error_count += 1 # Count as an error/skipped line for reporting

                    # --- Checkpointing ---
                    current_time = time.time()
                    # Checkpoint based on time interval or number of processed lines
                    if current_time - last_checkpoint_time >= 60 or self.processed_lines_count % 10000 == 0:
                        logger.info(f"Checkpointing at line index {current_line_index}...")
                        self._save_checkpoint(current_line_index)
                        self.vectorstore.persist()
                        last_checkpoint_time = current_time
                        # Reset count for *this run* after saving cumulative
                        self.total_lines_processed_cumulative += self.processed_lines_count
                        self.processed_lines_count = 0


            # Add any remaining documents
            if all_docs_batch:
                logger.info(f"Adding final batch of {len(all_docs_batch)} documents.")
                self._add_documents_to_vectorstore(all_docs_batch)

            # Final save and persist
            logger.info("JSONL processing finished. Saving final checkpoint and persisting vector store.")
            self._save_checkpoint(current_line_index) # Save progress up to the last line read
            self.vectorstore.persist()
            self.total_lines_processed_cumulative += self.processed_lines_count # Add final count

            end_time = time.time()
            logger.info(f"Successfully processed JSONL file.")
            logger.info(f"Lines read in this session: {lines_read_in_session}")
            logger.info(f"Lines successfully processed into chunks in this run: {self.processed_lines_count}")
            logger.info(f"Total lines processed cumulatively (across runs): {self.total_lines_processed_cumulative}")
            logger.info(f"Lines skipped/errored in this run: {error_count}")
            logger.info(f"Total chunks added to vector store in this run: {self.total_chunks_added - checkpoint_data.get('total_chunks_added', 0) if self.resume else self.total_chunks_added}")
            logger.info(f"Total time taken: {end_time - start_time:.2f} seconds")

        except FileNotFoundError:
             logger.critical(f"Fatal Error: Input JSONL file not found at {self.input_jsonl}")
             raise
        except Exception as e:
            logger.error(f"Error during JSONL processing around line index {current_line_index}: {e}", exc_info=True)
            logger.error("Attempting to save checkpoint before exiting due to error.")
            self._save_checkpoint(current_line_index)
            self.vectorstore.persist()
            raise # Re-raise the exception

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
                
                added_ids = self.vectorstore.add_documents(filtered_documents)
                self.total_chunks_added += len(added_ids)
                logger.info(f"Successfully added {len(added_ids)} documents to Chroma. Total chunks in store: ~{self.total_chunks_added}")
                return # Success
            except Exception as e:
                attempt += 1
                logger.error(f"Error adding documents to Chroma (Attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Failed to add batch to Chroma.")
                    return

def main():
    parser = argparse.ArgumentParser(description="Process research paper abstracts from a JSON Lines file, generate embeddings, and store in Chroma.")
    parser.add_argument("input_jsonl", help="Path to the input JSON Lines file.")
    parser.add_argument("output_dir", help="Directory to save the Chroma vector store and logs.")
    parser.add_argument("--text_keys", nargs='+', default=DEFAULT_TEXT_KEYS,
                        help=f"JSON keys containing the text to embed (default: {DEFAULT_TEXT_KEYS})")
    parser.add_argument("--meta_keys", nargs='+', default=DEFAULT_METADATA_KEYS,
                        help=f"JSON keys to include as metadata (default: {DEFAULT_METADATA_KEYS})")
    parser.add_argument("--id_key", default=DEFAULT_ID_KEY, help=f"JSON key for the unique paper identifier (default: {DEFAULT_ID_KEY})")
    parser.add_argument("--embed_model", default=DEFAULT_EMBEDDING_MODEL,
                        help=f"Hugging Face model name for embeddings (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Target size for text chunks (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk_overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                        help=f"Overlap between text chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Number of documents to batch add to Chroma (default: {DEFAULT_BATCH_SIZE})")
    # parser.add_argument("--embed_batch_size", type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE, help="Batch size for the embedding model itself") # Add if needed
    parser.add_argument("--shard_id", type=int, default=0, help="Shard ID for parallel processing (default: 0)")
    parser.add_argument("--total_shards", type=int, default=1, help="Total number of shards for parallel processing (default: 1)")
    # Keep the resume argument for backward compatibility but make it no-op
    parser.add_argument("--resume", action='store_true', help="Resume processing from the last checkpoint (now always active by default).")
    parser.add_argument("--force_restart", action='store_true', help="Force restart from beginning, ignoring any existing checkpoint.")
    parser.add_argument("--max_lines", type=int, default=None, help="Maximum number of lines to process (optional).")


    args = parser.parse_args()

    if not os.path.exists(args.input_jsonl):
        logger.critical(f"Input JSONL file not found: {args.input_jsonl}")
        return

    logger.info("Starting CS Paper JSONL Processing...")
    logger.info(f"Input JSONL: {args.input_jsonl}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Text Keys: {args.text_keys}")
    logger.info(f"Metadata Keys: {args.meta_keys}")
    logger.info(f"ID Key: {args.id_key}")
    logger.info(f"Embedding Model: {args.embed_model}")
    logger.info(f"Chunk Size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    logger.info(f"Chroma Batch Size: {args.batch_size}")
    logger.info(f"Sharding: {args.shard_id}/{args.total_shards}")
    logger.info(f"Max Lines: {args.max_lines if args.max_lines is not None else 'No limit'}")
    
    # If force_restart is specified, delete the checkpoint
    output_dir = args.output_dir
    if args.total_shards > 1:
        output_dir = os.path.join(output_dir, f"shard_{args.shard_id}")
    checkpoint_file = os.path.join(output_dir, "checkpoint_json.json")
    
    if args.force_restart and os.path.exists(checkpoint_file):
        logger.warning(f"--force_restart specified. Deleting existing checkpoint and starting from beginning.")
        try:
            os.remove(checkpoint_file)
        except Exception as e:
            logger.error(f"Error deleting checkpoint file: {e}")

    try:
        processor = CsPaperJsonProcessor(
            input_jsonl=args.input_jsonl,
            output_dir=args.output_dir,
            text_keys=args.text_keys,
            metadata_keys=args.meta_keys,
            embedding_model=args.embed_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            # embedding_batch_size=args.embed_batch_size, # Pass if added
            shard_id=args.shard_id,
            total_shards=args.total_shards,
            # resume flag is ignored now, always attempting to resume
            id_key=args.id_key,
            max_lines=args.max_lines
        )
        processor.process_jsonl()
        logger.info("Processing completed successfully.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        logger.critical("Processing failed.")

if __name__ == "__main__":
    main() 