"""
ARQ Worker Configuration.

This module defines the ARQ worker settings and background job functions
for processing marketing content asynchronously.

To run the worker:
    arq marketing_project.worker.WorkerSettings

Or with custom Redis:
    ARQ_REDIS_HOST=redis arq marketing_project.worker.WorkerSettings
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from arq import create_pool
from arq.connections import RedisSettings

from marketing_project.processors import (
    process_blog_post,
    process_release_notes,
    process_transcript,
)
from marketing_project.services.function_pipeline import FunctionPipeline
from marketing_project.services.internal_docs_scanner import get_internal_docs_scanner
from marketing_project.services.job_manager import JobStatus, get_job_manager

logger = logging.getLogger(__name__)


# ARQ Job Functions
async def process_blog_job(ctx, content_json: str, job_id: str, **kwargs) -> Dict:
    """
    Background job for processing blog posts.

    Args:
        ctx: ARQ context
        content_json: JSON string of blog content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Processing blog job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(job_id, 10, "Starting blog processing")

        # Store input content in job metadata
        try:
            content_dict = json.loads(content_json)
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata["input_content"] = content_dict
                # Also extract and store title for easier access
                if "title" in content_dict:
                    job.metadata["title"] = content_dict["title"]
                await job_manager._save_job_to_redis(job)
        except Exception as e:
            logger.warning(f"Failed to store input content for job {job_id}: {e}")

        # Process the blog post
        result_json = await process_blog_post(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(f"ARQ Worker: Blog job {job_id} completed successfully")

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Blog job {job_id} failed: {e}")
        raise


async def process_release_notes_job(
    ctx, content_json: str, job_id: str, **kwargs
) -> Dict:
    """
    Background job for processing release notes.

    Args:
        ctx: ARQ context
        content_json: JSON string of release notes content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Processing release notes job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Starting release notes processing"
        )

        # Store input content in job metadata
        try:
            content_dict = json.loads(content_json)
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata["input_content"] = content_dict
                # Also extract and store title for easier access
                if "title" in content_dict:
                    job.metadata["title"] = content_dict["title"]
                await job_manager._save_job_to_redis(job)
        except Exception as e:
            logger.warning(f"Failed to store input content for job {job_id}: {e}")

        # Process the release notes
        result_json = await process_release_notes(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(f"ARQ Worker: Release notes job {job_id} completed successfully")

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Release notes job {job_id} failed: {e}")
        raise


async def process_transcript_job(ctx, content_json: str, job_id: str, **kwargs) -> Dict:
    """
    Background job for processing transcripts.

    Args:
        ctx: ARQ context
        content_json: JSON string of transcript content
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters (e.g., metadata, _timeout)

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Processing transcript job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Starting transcript processing"
        )

        # Store input content in job metadata
        try:
            content_dict = json.loads(content_json)
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata["input_content"] = content_dict
                # Also extract and store title for easier access
                if "title" in content_dict:
                    job.metadata["title"] = content_dict["title"]
                await job_manager._save_job_to_redis(job)
        except Exception as e:
            logger.warning(f"Failed to store input content for job {job_id}: {e}")

        # Process the transcript
        result_json = await process_transcript(content_json, job_id=job_id)
        result_dict = json.loads(result_json)

        if result_dict.get("status") == "error":
            raise Exception(result_dict.get("message", "Processing failed"))

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(f"ARQ Worker: Transcript job {job_id} completed successfully")

        return result_dict

    except Exception as e:
        logger.error(f"ARQ Worker: Transcript job {job_id} failed: {e}")
        raise


async def analyze_design_kit_batch_job(
    ctx,
    content_batch: List[Dict[str, Any]],
    batch_index: int,
    parent_job_id: str,
    batch_job_id: str,
    **kwargs,
) -> Dict:
    """
    Analyze a batch of content pieces for design kit patterns.

    Args:
        ctx: ARQ context
        content_batch: List of content documents to analyze
        batch_index: Index of this batch (for logging)
        parent_job_id: Parent job ID
        batch_job_id: This batch job's ID

    Returns:
        Dictionary with batch analysis results
    """
    try:
        logger.info(
            f"ARQ Worker: Analyzing design kit batch {batch_index + 1} (job {batch_job_id}, parent {parent_job_id})"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            batch_job_id,
            0,
            f"Analyzing batch {batch_index + 1} ({len(content_batch)} pieces)",
        )

        import os

        from marketing_project.plugins.design_kit.tasks import DesignKitPlugin
        from marketing_project.services.function_pipeline import FunctionPipeline

        pipeline = FunctionPipeline(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7
        )

        plugin = DesignKitPlugin()
        analyses = []

        for idx, content_doc in enumerate(content_batch):
            progress = int((idx / len(content_batch)) * 90)  # 0-90%
            await job_manager.update_job_progress(
                batch_job_id,
                progress,
                f"Batch {batch_index + 1}: Analyzing {idx + 1}/{len(content_batch)}",
            )

            analysis = await plugin._analyze_content(
                pipeline, content_doc, idx, len(content_batch)
            )
            if analysis:
                analyses.append(analysis)

        await job_manager.update_job_progress(
            batch_job_id, 100, f"Batch {batch_index + 1} completed"
        )

        result = {
            "status": "success",
            "batch_index": batch_index,
            "analyses": analyses,
            "count": len(analyses),
        }

        logger.info(
            f"ARQ Worker: Batch {batch_index + 1} completed with {len(analyses)} analyses"
        )
        return result

    except Exception as e:
        logger.error(f"ARQ Worker: Batch {batch_index + 1} failed: {e}")
        raise


async def synthesize_design_kit_job(
    ctx,
    all_analyses: List[Dict[str, Any]],
    parent_job_id: str,
    synthesis_job_id: str,
    **kwargs,
) -> Dict:
    """
    Synthesize all content analyses into a final design kit config.

    Args:
        ctx: ARQ context
        all_analyses: List of all analysis results from batches
        parent_job_id: Parent job ID
        synthesis_job_id: This synthesis job's ID

    Returns:
        Dictionary with synthesized design kit config
    """
    try:
        logger.info(
            f"ARQ Worker: Synthesizing design kit config (job {synthesis_job_id}, parent {parent_job_id})"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            synthesis_job_id, 0, f"Synthesizing {len(all_analyses)} analyses"
        )

        import os

        from marketing_project.plugins.design_kit.tasks import DesignKitPlugin
        from marketing_project.services.function_pipeline import FunctionPipeline

        pipeline = FunctionPipeline(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.7
        )

        plugin = DesignKitPlugin()

        # Use the plugin's synthesis logic
        await job_manager.update_job_progress(
            synthesis_job_id, 50, "Calling AI to synthesize config"
        )

        # Flatten all analyses into a single list
        flat_analyses = []
        for batch_result in all_analyses:
            if isinstance(batch_result, dict) and "analyses" in batch_result:
                flat_analyses.extend(batch_result["analyses"])
            elif isinstance(batch_result, list):
                flat_analyses.extend(batch_result)
            else:
                flat_analyses.append(batch_result)

        # Generate config from analyses
        generated_config = await plugin._synthesize_config(
            pipeline, flat_analyses, synthesis_job_id
        )

        await job_manager.update_job_progress(
            synthesis_job_id, 100, "Synthesis completed"
        )

        result = {
            "status": "success",
            "config": generated_config.model_dump(mode="json"),
            "version": generated_config.version,
        }

        logger.info(f"ARQ Worker: Synthesis completed successfully")
        return result

    except Exception as e:
        logger.error(f"ARQ Worker: Synthesis failed: {e}")
        raise


async def refresh_design_kit_job(
    ctx, use_internal_docs: bool, job_id: str, **kwargs
) -> Dict:
    """
    Background job for refreshing design kit configuration using AI.

    This job orchestrates the process:
    1. Fetches all content from internal_docs
    2. Splits into batches of 20
    3. Creates parallel sub-jobs to analyze each batch
    4. Waits for all batches to complete
    5. Creates a synthesis job to combine all analyses
    6. Saves the final config

    Args:
        ctx: ARQ context
        use_internal_docs: Whether to enrich with internal docs configuration
        job_id: Job ID for tracking

    Returns:
        Dictionary with the generated config
    """
    try:
        logger.info(f"ARQ Worker: Refreshing design kit configuration (job {job_id})")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 5, "Starting design kit generation"
        )

        # Import here to avoid circular imports
        import uuid

        from marketing_project.plugins.design_kit.tasks import DesignKitPlugin
        from marketing_project.services.design_kit_manager import get_design_kit_manager
        from marketing_project.services.scanned_document_db import (
            get_scanned_document_db,
        )

        manager = await get_design_kit_manager()

        content_batches = []
        batch_job_ids = []

        if use_internal_docs:
            await job_manager.update_job_progress(
                job_id, 10, "Fetching content from internal_docs"
            )
            logger.info(f"Design kit job {job_id}: Fetching content from internal_docs")

            try:
                db = get_scanned_document_db()
                all_docs = db.get_all_active_documents()

                logger.info(
                    f"Found {len(all_docs)} total active documents in internal_docs"
                )

                # Log detailed info about all documents for debugging
                if all_docs:
                    logger.info(f"Sample of first 3 documents structure:")
                    for i, doc in enumerate(all_docs[:3]):
                        logger.info(
                            f"  Doc {i+1}: title='{doc.title[:60]}', "
                            f"url='{doc.url[:60]}', "
                            f"has_metadata={doc.metadata is not None}, "
                            f"content_text_len={len(doc.metadata.content_text) if doc.metadata and doc.metadata.content_text else 0}, "
                            f"content_summary_len={len(doc.metadata.content_summary) if doc.metadata and doc.metadata.content_summary else 0}, "
                            f"headings_count={len(doc.metadata.headings) if doc.metadata and doc.metadata.headings else 0}, "
                            f"word_count={doc.metadata.word_count if doc.metadata else None}"
                        )

                # Filter to only content that has substantial text content
                # Accept documents with content_text OR content_summary (fallback)
                # Lowered threshold to 50 chars to be more inclusive
                content_docs = []
                filtered_reasons = {
                    "no_metadata": 0,
                    "no_content": 0,
                    "too_short": 0,
                    "accepted": 0,
                }

                for doc in all_docs:
                    # Check if metadata exists
                    if not doc.metadata:
                        filtered_reasons["no_metadata"] += 1
                        continue

                    # Check primary content_text field (lowered threshold to 50)
                    if (
                        doc.metadata.content_text
                        and len(doc.metadata.content_text.strip()) >= 50
                    ):
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Fallback: use content_summary if content_text is missing/short
                    if (
                        doc.metadata.content_summary
                        and len(doc.metadata.content_summary.strip()) >= 50
                    ):
                        # Copy summary to content_text for analysis
                        doc.metadata.content_text = doc.metadata.content_summary
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Fallback: use title + headings if available
                    if (
                        doc.title
                        and doc.metadata.headings
                        and len(doc.metadata.headings) > 0
                    ):
                        # Create minimal content from title and headings
                        minimal_content = f"{doc.title}\n\n" + "\n".join(
                            doc.metadata.headings[:10]
                        )
                        if len(minimal_content.strip()) >= 50:
                            doc.metadata.content_text = minimal_content
                            content_docs.append(doc)
                            filtered_reasons["accepted"] += 1
                            continue

                    # Fallback: use title alone if it's substantial
                    if doc.title and len(doc.title.strip()) >= 50:
                        doc.metadata.content_text = doc.title
                        content_docs.append(doc)
                        filtered_reasons["accepted"] += 1
                        continue

                    # Document was filtered out
                    if (
                        not doc.metadata.content_text
                        and not doc.metadata.content_summary
                    ):
                        filtered_reasons["no_content"] += 1
                    else:
                        filtered_reasons["too_short"] += 1

                # Log filtering statistics
                logger.info(
                    f"Content filtering results: {filtered_reasons['accepted']} accepted, "
                    f"{filtered_reasons['no_metadata']} no metadata, "
                    f"{filtered_reasons['no_content']} no content, "
                    f"{filtered_reasons['too_short']} too short"
                )

                if not content_docs and all_docs:
                    # Log detailed info about why documents were filtered out
                    logger.warning(
                        f"All {len(all_docs)} documents filtered out. Filtering breakdown: {filtered_reasons}"
                    )
                    # Show detailed info about first few documents
                    for i, doc in enumerate(all_docs[:5]):
                        logger.warning(
                            f"  Filtered doc {i+1}: title='{doc.title[:80]}', "
                            f"has_metadata={doc.metadata is not None}, "
                            f"content_text={bool(doc.metadata.content_text) if doc.metadata else False} "
                            f"(len={len(doc.metadata.content_text.strip()) if doc.metadata and doc.metadata.content_text else 0}), "
                            f"content_summary={bool(doc.metadata.content_summary) if doc.metadata else False} "
                            f"(len={len(doc.metadata.content_summary.strip()) if doc.metadata and doc.metadata.content_summary else 0}), "
                            f"headings={len(doc.metadata.headings) if doc.metadata and doc.metadata.headings else 0}"
                        )

                if content_docs:
                    # Split into batches of 20
                    BATCH_SIZE = 20
                    for i in range(0, len(content_docs), BATCH_SIZE):
                        batch = content_docs[i : i + BATCH_SIZE]
                        content_batches.append(
                            [doc.model_dump(mode="json") for doc in batch]
                        )

                    logger.info(
                        f"Split {len(content_docs)} content pieces into {len(content_batches)} batches "
                        f"of up to {BATCH_SIZE} pieces each"
                    )
                else:
                    logger.warning(
                        f"No content found in internal_docs (checked {len(all_docs)} documents)"
                    )
            except Exception as e:
                logger.warning(f"Error fetching content from internal_docs: {e}")

        if content_batches:
            # Create and submit batch analysis jobs
            await job_manager.update_job_progress(
                job_id, 15, f"Creating {len(content_batches)} analysis batches"
            )

            for batch_idx, batch in enumerate(content_batches):
                batch_job_id = str(uuid.uuid4())
                batch_job_ids.append(batch_job_id)

                # Create job for this batch
                batch_job = await job_manager.create_job(
                    job_type="design_kit_batch_analysis",
                    content_id=f"batch_{batch_idx}",
                    metadata={
                        "parent_job_id": job_id,
                        "batch_index": batch_idx,
                        "batch_size": len(batch),
                    },
                    job_id=batch_job_id,
                )

                # Submit batch analysis job via JobManager to properly track arq_job_id
                arq_batch_job_id = await job_manager.submit_to_arq(
                    batch_job_id,
                    "analyze_design_kit_batch_job",
                    batch,
                    batch_idx,
                    job_id,
                    batch_job_id,
                    _timeout=600,  # 10 minutes per batch
                )

                logger.info(
                    f"Submitted batch {batch_idx + 1}/{len(content_batches)} for analysis (job {batch_job_id})"
                )

            # Wait for all batch jobs to complete
            await job_manager.update_job_progress(
                job_id, 20, f"Waiting for {len(batch_job_ids)} batches to complete"
            )
            logger.info(f"Waiting for {len(batch_job_ids)} batch jobs to complete...")

            all_analyses = []
            completed_batches = 0

            # Poll for batch completion
            import asyncio

            max_wait_time = 600 * len(batch_job_ids)  # 10 min per batch
            start_time = asyncio.get_event_loop().time()

            while completed_batches < len(batch_job_ids):
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    raise TimeoutError(f"Batch jobs timed out after {max_wait_time}s")

                for batch_job_id in batch_job_ids:
                    batch_job = await job_manager.get_job(batch_job_id)
                    if batch_job:
                        if batch_job.status.value == "completed":
                            if (
                                batch_job.result
                                and batch_job.result.get("status") == "success"
                            ):
                                all_analyses.append(batch_job.result)
                                completed_batches += 1
                                batch_job_ids.remove(batch_job_id)

                                progress = 20 + int(
                                    (completed_batches / len(content_batches)) * 50
                                )  # 20-70%
                                await job_manager.update_job_progress(
                                    job_id,
                                    progress,
                                    f"Completed {completed_batches}/{len(content_batches)} batches",
                                )
                                logger.info(
                                    f"Batch {completed_batches}/{len(content_batches)} completed"
                                )
                        elif batch_job.status.value == "failed":
                            logger.error(
                                f"Batch job {batch_job_id} failed: {batch_job.error}"
                            )
                            # Continue with other batches
                            batch_job_ids.remove(batch_job_id)
                            completed_batches += 1

                if completed_batches < len(content_batches):
                    await asyncio.sleep(2)  # Poll every 2 seconds

            logger.info(
                f"All {len(content_batches)} batches completed. Total analyses: {sum(len(a.get('analyses', [])) for a in all_analyses)}"
            )

            # Create synthesis job
            await job_manager.update_job_progress(
                job_id, 75, "Synthesizing all analyses"
            )
            synthesis_job_id = str(uuid.uuid4())

            synthesis_job = await job_manager.create_job(
                job_type="design_kit_synthesis",
                content_id="synthesis",
                metadata={"parent_job_id": job_id, "analysis_count": len(all_analyses)},
                job_id=synthesis_job_id,
            )

            # Submit synthesis job via JobManager to properly track arq_job_id
            arq_synthesis_job_id = await job_manager.submit_to_arq(
                synthesis_job_id,
                "synthesize_design_kit_job",
                all_analyses,
                job_id,
                synthesis_job_id,
                _timeout=600,  # 10 minutes for synthesis
            )

            logger.info(f"Submitted synthesis job {synthesis_job_id}")

            # Wait for synthesis to complete
            await job_manager.update_job_progress(
                job_id, 80, "Waiting for synthesis to complete"
            )

            synthesis_complete = False
            max_synthesis_wait = 600  # 10 minutes
            synthesis_start = asyncio.get_event_loop().time()

            while not synthesis_complete:
                if (
                    asyncio.get_event_loop().time() - synthesis_start
                    > max_synthesis_wait
                ):
                    raise TimeoutError("Synthesis job timed out")

                synth_job = await job_manager.get_job(synthesis_job_id)
                if synth_job:
                    if synth_job.status.value == "completed":
                        if (
                            synth_job.result
                            and synth_job.result.get("status") == "success"
                        ):
                            generated_config_dict = synth_job.result.get("config")
                            if generated_config_dict:
                                from marketing_project.models.design_kit_config import (
                                    DesignKitConfig,
                                )

                                generated_config = DesignKitConfig(
                                    **generated_config_dict
                                )
                                synthesis_complete = True
                            else:
                                raise Exception(
                                    "Synthesis job completed but no config in result"
                                )
                        else:
                            raise Exception(f"Synthesis job failed: {synth_job.result}")
                    elif synth_job.status.value == "failed":
                        raise Exception(f"Synthesis job failed: {synth_job.error}")

                await asyncio.sleep(2)

            logger.info("Synthesis completed successfully")
        else:
            # No content batches - generate generic config
            await job_manager.update_job_progress(
                job_id, 30, "Generating generic config (no content found)"
            )
            logger.info("No content batches, generating generic config")
            generated_config = await manager.generate_config_with_ai(
                use_internal_docs=False, job_id=job_id
            )

        # Set metadata and enrich
        generated_config.version = "1.0.0"
        generated_config.created_at = datetime.utcnow()
        generated_config.updated_at = datetime.utcnow()
        generated_config.is_active = True

        if use_internal_docs:
            await job_manager.update_job_progress(
                job_id, 90, "Enriching with internal docs"
            )
            await manager._enrich_with_internal_docs(generated_config)

        await job_manager.update_job_progress(job_id, 95, "Saving configuration")

        # Save the generated config
        success = await manager.save_config(generated_config, set_active=True)

        if not success:
            raise Exception("Failed to save generated design kit configuration")

        await job_manager.update_job_progress(job_id, 100, "Completed")

        # Return the config as dict
        result = {
            "status": "success",
            "config": generated_config.model_dump(mode="json"),
            "version": generated_config.version,
        }

        logger.info(
            f"ARQ Worker: Design kit refresh job {job_id} completed successfully"
        )
        return result

    except Exception as e:
        logger.error(
            f"ARQ Worker: Design kit refresh job {job_id} failed: {e}", exc_info=True
        )
        raise


async def resume_pipeline_job(
    ctx, original_job_id: str, context_data: Dict, job_id: str
) -> Dict:
    """
    Background job for resuming a pipeline after approval.

    Args:
        ctx: ARQ context
        original_job_id: ID of the original job that was waiting for approval
        context_data: Saved pipeline context from approval_manager
        job_id: New job ID for the resume job

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Resuming pipeline for original job {original_job_id}, new job {job_id}"
        )

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 10, "Resuming pipeline after approval"
        )

        # Store input content in job metadata if available
        input_content = context_data.get("input_content") or context_data.get(
            "original_content"
        )
        if input_content:
            job = await job_manager.get_job(job_id)
            if job:
                job.metadata["input_content"] = input_content
                # Also extract and store title for easier access
                if isinstance(input_content, dict) and "title" in input_content:
                    job.metadata["title"] = input_content["title"]
                await job_manager._save_job_to_redis(job)

        # Import pipeline
        from marketing_project.services.function_pipeline import FunctionPipeline

        # Create pipeline and resume
        pipeline = FunctionPipeline(model="gpt-4o-mini", temperature=0.7)

        # Determine content type from context
        content_type = context_data.get("content_type", "blog_post")

        # Resume pipeline
        result = await pipeline.resume_pipeline(
            context_data=context_data, job_id=job_id, content_type=content_type
        )

        pipeline_status = result.get("pipeline_status")

        if pipeline_status == "failed":
            raise Exception(result.get("error", "Resume pipeline failed"))

        if pipeline_status == "waiting_for_approval":
            # Pipeline stopped for approval - job is already marked as WAITING_FOR_APPROVAL
            # by resume_pipeline, just return the result
            logger.info(
                f"ARQ Worker: Resume job {job_id} stopped for approval at step "
                f"{result.get('metadata', {}).get('approval_step_name', 'unknown')}"
            )
            return {
                "status": "waiting_for_approval",
                "original_job_id": original_job_id,
                "resume_job_id": job_id,
                "result": result,
                "message": f"Pipeline waiting for approval at step {result.get('metadata', {}).get('approval_step_name', 'unknown')}",
            }

        # Pipeline completed successfully
        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(f"ARQ Worker: Resume job {job_id} completed successfully")

        return {
            "status": "success",
            "original_job_id": original_job_id,
            "resume_job_id": job_id,
            "result": result,
            "message": "Pipeline resumed successfully",
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Resume job {job_id} failed: {e}")
        raise


async def retry_step_job(
    ctx,
    step_name: str,
    input_data: Dict,
    context: Dict,
    job_id: str,
    approval_id: str,
    user_guidance: Optional[str] = None,
) -> Dict:
    """
    Background job for retrying a single pipeline step.

    Args:
        ctx: ARQ context
        step_name: Name of the step to retry (e.g., "seo_keywords", "marketing_brief")
        input_data: Input data for the step
        context: Context from previous pipeline steps
        job_id: Job ID for tracking
        approval_id: Approval ID that triggered the retry
        user_guidance: Optional user guidance/feedback for regeneration

    Returns:
        Step execution result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Retrying step '{step_name}' for job {job_id} (approval: {approval_id})"
        )
        if user_guidance:
            logger.info(
                f"ARQ Worker: Using user guidance for retry: {user_guidance[:100]}..."
            )

        # Import retry service
        from marketing_project.services.step_retry_service import get_retry_service

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_progress(job_id, 10, f"Retrying step: {step_name}")

        # Get retry service and execute the step
        retry_service = get_retry_service()
        result = await retry_service.retry_step(
            step_name=step_name,
            input_data=input_data,
            context=context,
            job_id=job_id,
            user_guidance=user_guidance,
        )

        if result["status"] == "error":
            raise Exception(result.get("error_message", "Step retry failed"))

        await job_manager.update_job_progress(
            job_id, 100, f"Step '{step_name}' retry completed"
        )
        logger.info(
            f"ARQ Worker: Step '{step_name}' retry completed successfully "
            f"for job {job_id} (approval: {approval_id})"
        )

        return {
            "status": "success",
            "approval_id": approval_id,
            "step_name": step_name,
            "result": result,
            "message": f"Step '{step_name}' retried successfully",
        }

    except Exception as e:
        logger.error(
            f"ARQ Worker: Step '{step_name}' retry failed for job {job_id}: {e}"
        )
        raise


async def execute_single_step_job(
    ctx,
    step_name: str,
    content_json: str,
    context: Dict[str, Any],
    job_id: str,
    **kwargs,
) -> Dict:
    """
    Background job for executing a single pipeline step independently.

    Args:
        ctx: ARQ context
        step_name: Name of the step to execute (e.g., "seo_keywords")
        content_json: JSON string of input content
        context: Dictionary containing all required context keys for the step
        job_id: Job ID for tracking
        **kwargs: Additional ARQ-specific parameters

    Returns:
        Step execution result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Executing single step '{step_name}' for job {job_id}")

        # Update job manager
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        await job_manager.update_job_progress(
            job_id, 10, f"Executing step: {step_name}"
        )

        # Create pipeline instance
        pipeline = FunctionPipeline()

        # Execute the step
        result = await pipeline.execute_single_step(
            step_name=step_name,
            content_json=content_json,
            context=context,
            job_id=job_id,
        )

        # Update job with result
        await job_manager.update_job_progress(
            job_id, 90, f"Step '{step_name}' completed"
        )
        await job_manager.update_job_status(job_id, JobStatus.COMPLETED)

        # Store result in job
        job = await job_manager.get_job(job_id)
        if job:
            job.result = result
            job.current_step = step_name
            await job_manager._save_job_to_redis(job)

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Single step '{step_name}' execution completed successfully for job {job_id}"
        )

        return {
            "status": "success",
            "step_name": step_name,
            "result": result,
            "message": f"Step '{step_name}' executed successfully",
        }

    except Exception as e:
        logger.error(
            f"ARQ Worker: Single step '{step_name}' execution failed for job {job_id}: {e}"
        )
        job_manager = get_job_manager()
        await job_manager.update_job_status(job_id, JobStatus.FAILED)
        await job_manager.update_job_progress(
            job_id, 0, f"Step execution failed: {str(e)}"
        )
        job = await job_manager.get_job(job_id)
        if job:
            job.error = str(e)
            await job_manager._save_job_to_redis(job)
        raise


async def bulk_rescan_documents_job(ctx, urls: List[str], job_id: str) -> Dict:
    """
    Background job for bulk re-scanning documents.

    Args:
        ctx: ARQ context
        urls: List of URLs to re-scan
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Bulk re-scanning {len(urls)} documents for job {job_id}"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting bulk re-scan of {len(urls)} documents"
        )

        scanner = await get_internal_docs_scanner()
        scanned_count = 0
        failed_count = 0
        errors = []

        for idx, url in enumerate(urls):
            try:
                await scanner._scan_single_url(url, save_to_db=True)
                scanned_count += 1
                progress = int((idx + 1) / len(urls) * 100)
                await job_manager.update_job_progress(
                    job_id, progress, f"Scanned {idx + 1}/{len(urls)} documents"
                )
            except Exception as e:
                failed_count += 1
                errors.append(f"{url}: {str(e)}")
                logger.warning(f"Failed to scan {url}: {e}")

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Bulk re-scan job {job_id} completed: {scanned_count} succeeded, {failed_count} failed"
        )

        return {
            "status": "success",
            "scanned_count": scanned_count,
            "failed_count": failed_count,
            "errors": errors[:20],  # Limit errors
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Bulk re-scan job {job_id} failed: {e}")
        raise


async def scan_from_url_job(
    ctx,
    base_url: str,
    max_depth: int,
    follow_external: bool,
    max_pages: int,
    merge_with_existing: bool,
    job_id: str,
) -> Dict:
    """
    Background job for scanning documents from a base URL.

    Args:
        ctx: ARQ context
        base_url: Base URL to start crawling from
        max_depth: Maximum crawl depth
        follow_external: Whether to follow external links
        max_pages: Maximum number of pages to scan
        merge_with_existing: Whether to merge with existing config
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(f"ARQ Worker: Scanning from base URL {base_url} for job {job_id}")

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting scan from {base_url}"
        )

        scanner = await get_internal_docs_scanner()
        scanned_docs = await scanner.scan_from_base_url(
            base_url=base_url,
            max_depth=max_depth,
            follow_external=follow_external,
            max_pages=max_pages,
        )

        await job_manager.update_job_progress(
            job_id, 50, f"Scanned {len(scanned_docs)} documents, merging with config..."
        )

        if merge_with_existing:
            from datetime import datetime

            from marketing_project.models.internal_docs_config import InternalDocsConfig
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            try:
                manager = await get_internal_docs_manager()
                config = await manager.get_active_config()

                if config:
                    # Update existing config
                    logger.info(
                        f"Updating existing config with {len(scanned_docs)} scanned documents"
                    )
                    config = await manager.merge_scan_results(config, scanned_docs)
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save updated config for job {job_id}")
                        raise Exception("Failed to save updated config")
                    logger.info(
                        f"Successfully updated config with {len(config.scanned_documents)} documents"
                    )
                else:
                    # Create new config if none exists (save_config will handle versioning)
                    logger.info(
                        f"Creating new config with {len(scanned_docs)} scanned documents"
                    )
                    config = InternalDocsConfig(
                        scanned_documents=scanned_docs,
                        version="1.0.0",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save new config for job {job_id}")
                        raise Exception("Failed to save new config")
                    logger.info(
                        f"Successfully created new config with {len(scanned_docs)} documents"
                    )
            except Exception as e:
                logger.error(
                    f"Error saving config for job {job_id}: {e}", exc_info=True
                )
                raise
        else:
            logger.info(
                f"merge_with_existing=False, skipping config save for job {job_id}"
            )

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Scan from URL job {job_id} completed: {len(scanned_docs)} documents scanned"
        )

        return {
            "status": "success",
            "message": f"Scanned {len(scanned_docs)} documents from {base_url}",
            "scanned_count": len(scanned_docs),
            "scanned_documents": [doc.model_dump(mode="json") for doc in scanned_docs],
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Scan from URL job {job_id} failed: {e}")
        raise


async def scan_from_list_job(
    ctx, urls: List[str], merge_with_existing: bool, job_id: str
) -> Dict:
    """
    Background job for scanning documents from a list of URLs.

    Args:
        ctx: ARQ context
        urls: List of URLs to scan
        merge_with_existing: Whether to merge with existing config
        job_id: Job ID for tracking

    Returns:
        Processing result dictionary
    """
    try:
        logger.info(
            f"ARQ Worker: Scanning from URL list ({len(urls)} URLs) for job {job_id}"
        )

        job_manager = get_job_manager()
        await job_manager.update_job_progress(
            job_id, 0, f"Starting scan from {len(urls)} URLs"
        )

        scanner = await get_internal_docs_scanner()
        scanned_docs = await scanner.scan_from_url_list(urls=urls)

        await job_manager.update_job_progress(
            job_id, 50, f"Scanned {len(scanned_docs)} documents, merging with config..."
        )

        if merge_with_existing:
            from datetime import datetime

            from marketing_project.models.internal_docs_config import InternalDocsConfig
            from marketing_project.services.internal_docs_manager import (
                get_internal_docs_manager,
            )

            try:
                manager = await get_internal_docs_manager()
                config = await manager.get_active_config()

                if config:
                    # Update existing config
                    logger.info(
                        f"Updating existing config with {len(scanned_docs)} scanned documents"
                    )
                    config = await manager.merge_scan_results(config, scanned_docs)
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save updated config for job {job_id}")
                        raise Exception("Failed to save updated config")
                    logger.info(
                        f"Successfully updated config with {len(config.scanned_documents)} documents"
                    )
                else:
                    # Create new config if none exists (save_config will handle versioning)
                    logger.info(
                        f"Creating new config with {len(scanned_docs)} scanned documents"
                    )
                    config = InternalDocsConfig(
                        scanned_documents=scanned_docs,
                        version="1.0.0",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    success = await manager.save_config(config, set_active=True)
                    if not success:
                        logger.error(f"Failed to save new config for job {job_id}")
                        raise Exception("Failed to save new config")
                    logger.info(
                        f"Successfully created new config with {len(scanned_docs)} documents"
                    )
            except Exception as e:
                logger.error(
                    f"Error saving config for job {job_id}: {e}", exc_info=True
                )
                raise
        else:
            logger.info(
                f"merge_with_existing=False, skipping config save for job {job_id}"
            )

        await job_manager.update_job_progress(job_id, 100, "Completed")
        logger.info(
            f"ARQ Worker: Scan from list job {job_id} completed: {len(scanned_docs)} documents scanned"
        )

        return {
            "status": "success",
            "message": f"Scanned {len(scanned_docs)} documents from {len(urls)} URLs",
            "scanned_count": len(scanned_docs),
            "scanned_documents": [doc.model_dump(mode="json") for doc in scanned_docs],
        }

    except Exception as e:
        logger.error(f"ARQ Worker: Scan from list job {job_id} failed: {e}")
        raise


# ARQ Worker startup/shutdown hooks
async def startup(ctx):
    """Called when worker starts."""
    logger.info("ARQ Worker starting up...")
    logger.info(f"Redis connection: {ctx['redis']}")


async def shutdown(ctx):
    """Called when worker shuts down."""
    logger.info("ARQ Worker shutting down...")


# ARQ Worker Settings
class WorkerSettings:
    """
    ARQ Worker configuration.

    NOTE: ARQ manages its own Redis connections and cannot share the RedisManager
    connection pool. ARQ requires RedisSettings for its internal connection management.
    However, we use the same environment variables to ensure consistency.

    Environment Variables:
        REDIS_HOST: Redis host (default: localhost)
        REDIS_PORT: Redis port (default: 6379)
        REDIS_DATABASE: Redis database number (default: 0)
        REDIS_PASSWORD: Redis password (optional)
        ARQ_MAX_JOBS: Maximum concurrent jobs (default: 10)
        ARQ_JOB_TIMEOUT: Job timeout in seconds (default: 1800 = 30 minutes)
    """

    # Redis connection settings
    # ARQ uses its own connection management, but we use the same env vars for consistency
    # SSL is not supported
    redis_settings = RedisSettings(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        database=int(os.getenv("REDIS_DATABASE", "0")),
        password=os.getenv("REDIS_PASSWORD"),
    )

    # Job functions to register
    # ARQ will use the function names automatically
    functions = [
        process_blog_job,
        process_release_notes_job,
        process_transcript_job,
        resume_pipeline_job,
        retry_step_job,
        execute_single_step_job,
        bulk_rescan_documents_job,
        scan_from_url_job,
        scan_from_list_job,
        refresh_design_kit_job,
        analyze_design_kit_batch_job,
        synthesize_design_kit_job,
    ]

    # Worker settings
    max_jobs = int(os.getenv("ARQ_MAX_JOBS", "10"))  # Max concurrent jobs
    # Default timeout: 30 minutes (1800s) to support long-running design kit jobs
    # Design kit jobs can take 20-30 minutes due to multiple batch analysis jobs + synthesis
    job_timeout = int(os.getenv("ARQ_JOB_TIMEOUT", "1800"))  # 30 minutes default
    keep_result = 3600  # Keep results for 1 hour
    keep_result_forever = False

    # Startup and shutdown hooks
    on_startup = startup
    on_shutdown = shutdown

    # Worker name
    name = "marketing-worker"

    # Health check interval (seconds)
    health_check_interval = 60

    # Log all jobs
    log_results = True


# Helper function to get ARQ pool (for API to enqueue jobs)
async def get_arq_pool():
    """
    Get ARQ Redis pool for enqueueing jobs.

    NOTE: This creates a separate ARQ-managed connection pool. ARQ cannot use
    the RedisManager connection pool because ARQ requires its own connection
    management for job queuing, result storage, and worker coordination.

    Both ARQ and RedisManager use the same environment variables for consistency.
    """
    redis_settings = WorkerSettings.redis_settings
    return await create_pool(redis_settings)
