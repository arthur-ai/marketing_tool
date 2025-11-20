"""
Transcript Preprocessing Approval plugin tasks for Marketing Project.

This plugin handles validation and approval of transcript preprocessing data
before proceeding to SEO keywords extraction.
"""

import logging
from typing import Any, Dict, Optional

from marketing_project.models.pipeline_steps import (
    TranscriptContentExtractionResult,
    TranscriptDurationExtractionResult,
    TranscriptPreprocessingApprovalResult,
    TranscriptSpeakersExtractionResult,
)
from marketing_project.plugins.base import PipelineStepPlugin

logger = logging.getLogger(
    "marketing_project.plugins.transcript_preprocessing_approval"
)


class TranscriptPreprocessingApprovalPlugin(PipelineStepPlugin):
    """Plugin for Transcript Preprocessing Approval step."""

    @property
    def step_name(self) -> str:
        return "transcript_preprocessing_approval"

    @property
    def step_number(self) -> int:
        return 1

    @property
    def response_model(self) -> type[TranscriptPreprocessingApprovalResult]:
        return TranscriptPreprocessingApprovalResult

    def get_required_context_keys(self) -> list[str]:
        return ["input_content"]

    def _build_prompt_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build prompt context for transcript preprocessing approval step.

        Extracts transcript-specific fields for validation.
        """
        prompt_context = super()._build_prompt_context(context)

        # Get input content
        input_content = context.get("input_content", {})
        if isinstance(input_content, dict):
            # Extract transcript fields
            prompt_context["transcript_id"] = input_content.get("id", "N/A")
            prompt_context["transcript_title"] = input_content.get("title", "N/A")
            prompt_context["transcript_content"] = input_content.get("content", "")
            prompt_context["transcript_snippet"] = input_content.get("snippet", "")
            prompt_context["transcript_speakers"] = input_content.get("speakers", [])
            prompt_context["transcript_duration"] = input_content.get("duration")
            prompt_context["transcript_type"] = input_content.get("transcript_type")
            prompt_context["transcript_metadata"] = input_content.get("metadata", {})

            # Include parsing information if available
            prompt_context["parsing_confidence"] = input_content.get(
                "parsing_confidence"
            )
            prompt_context["detected_format"] = input_content.get("detected_format")
            prompt_context["parsing_warnings"] = input_content.get(
                "parsing_warnings", []
            )
            prompt_context["quality_metrics"] = input_content.get("quality_metrics", {})
            prompt_context["speaking_time_per_speaker"] = input_content.get(
                "speaking_time_per_speaker", {}
            )
            prompt_context["detected_language"] = input_content.get("detected_language")
            prompt_context["key_topics"] = input_content.get("key_topics", [])
            prompt_context["conversation_flow"] = input_content.get(
                "conversation_flow", {}
            )

            # Create content summary (first 500 chars)
            content_str = prompt_context.get("transcript_content", "")
            if content_str:
                prompt_context["content_summary"] = (
                    content_str[:500] + "..." if len(content_str) > 500 else content_str
                )
            else:
                prompt_context["content_summary"] = "No content available"

        # Add content type
        prompt_context["content_type"] = context.get("content_type", "blog_post")

        return prompt_context

    async def execute(
        self,
        context: Dict[str, Any],
        pipeline: Any,
        job_id: Optional[str] = None,
    ) -> TranscriptPreprocessingApprovalResult:
        """
        Execute transcript preprocessing approval step.

        This step validates transcript fields and requires approval if issues are found.
        Only runs when content_type is "transcript".

        Args:
            context: Context containing input_content
            pipeline: FunctionPipeline instance
            job_id: Optional job ID for tracking

        Returns:
            TranscriptPreprocessingApprovalResult with validation status
        """
        # Check if this is transcript content - if not, skip validation
        content_type = context.get("content_type", "blog_post")
        if content_type != "transcript":
            logger.info(
                f"Skipping transcript preprocessing approval for content_type: {content_type}"
            )
            # Return a default valid result for non-transcript content
            return TranscriptPreprocessingApprovalResult(
                is_valid=True,
                speakers_validated=True,
                duration_validated=True,
                content_validated=True,
                transcript_type_validated=True,
                validation_issues=[],
                speakers=[],
                duration=None,
                transcript_type=None,
                content_summary=None,
                confidence_score=1.0,
                requires_approval=False,
                approval_suggestions=[],
            )

        logger.info(
            "Executing transcript preprocessing approval step (3 sequential LLM calls)"
        )

        # Build prompt context for all calls
        prompt_context = self._build_prompt_context(context)

        # Get execution step number from context if available (for approval tracking)
        execution_step_number = context.get("_execution_step_number", self.step_number)

        # ========================================
        # Call 1: Content Extraction
        # ========================================
        logger.info("Step 1/3: Extracting transcript content")
        content_result = None
        try:
            content_prompt = pipeline._get_user_prompt(
                "transcript_content_extraction", prompt_context
            )
            content_system_instruction = pipeline._get_system_instruction(
                "transcript_content_extraction", prompt_context
            )

            # Skip approval on individual calls - will check on cumulative result
            content_result = await pipeline._call_function(
                prompt=content_prompt,
                system_instruction=content_system_instruction,
                response_model=TranscriptContentExtractionResult,
                step_name="transcript_content_extraction",  # Internal step name, approval skipped
                step_number=execution_step_number,
                context=context,
                job_id=None,  # Skip approval check for individual calls
            )

            # Log success/failure for content extraction
            if isinstance(content_result, TranscriptContentExtractionResult):
                prompt_context["extracted_content"] = content_result.extracted_content
                confidence_str = (
                    f"{content_result.confidence_score:.2f}"
                    if content_result.confidence_score is not None
                    else "N/A"
                )
                if content_result.content_validated:
                    logger.info(
                        f"✓ Step 1/3 SUCCESS: Content extraction completed - "
                        f"{len(content_result.extracted_content)} characters extracted, "
                        f"validated={content_result.content_validated}, "
                        f"confidence={confidence_str}, "
                        f"transcript_type={content_result.transcript_type or 'N/A'}"
                    )
                else:
                    logger.warning(
                        f"⚠ Step 1/3 PARTIAL: Content extraction completed but validation failed - "
                        f"{len(content_result.extracted_content)} characters extracted, "
                        f"validation_issues={len(content_result.validation_issues)}, "
                        f"confidence={confidence_str}"
                    )
                    if content_result.validation_issues:
                        logger.warning(
                            f"  Validation issues: {', '.join(content_result.validation_issues)}"
                        )
            else:
                logger.error(
                    f"✗ Step 1/3 FAILED: Content extraction returned invalid result type: {type(content_result)}"
                )
        except Exception as e:
            logger.error(
                f"✗ Step 1/3 FAILED: Content extraction raised exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise  # Re-raise to stop pipeline if content extraction fails

        # ========================================
        # Call 2: Speakers Extraction
        # ========================================
        logger.info("Step 2/3: Extracting speakers")
        speakers_result = None
        try:
            speakers_prompt = pipeline._get_user_prompt(
                "transcript_speakers_extraction", prompt_context
            )
            speakers_system_instruction = pipeline._get_system_instruction(
                "transcript_speakers_extraction", prompt_context
            )

            # Skip approval on individual calls - will check on cumulative result
            speakers_result = await pipeline._call_function(
                prompt=speakers_prompt,
                system_instruction=speakers_system_instruction,
                response_model=TranscriptSpeakersExtractionResult,
                step_name="transcript_speakers_extraction",  # Internal step name, approval skipped
                step_number=execution_step_number,
                context=context,
                job_id=None,  # Skip approval check for individual calls
            )

            # Log success/failure for speakers extraction
            if isinstance(speakers_result, TranscriptSpeakersExtractionResult):
                confidence_str = (
                    f"{speakers_result.confidence_score:.2f}"
                    if speakers_result.confidence_score is not None
                    else "N/A"
                )
                if speakers_result.speakers_validated:
                    logger.info(
                        f"✓ Step 2/3 SUCCESS: Speakers extraction completed - "
                        f"speakers={speakers_result.speakers}, "
                        f"count={len(speakers_result.speakers)}, "
                        f"validated={speakers_result.speakers_validated}, "
                        f"confidence={confidence_str}"
                    )
                else:
                    logger.warning(
                        f"⚠ Step 2/3 PARTIAL: Speakers extraction completed but validation failed - "
                        f"speakers={speakers_result.speakers}, "
                        f"count={len(speakers_result.speakers)}, "
                        f"validation_issues={len(speakers_result.validation_issues)}, "
                        f"confidence={confidence_str}"
                    )
                    if speakers_result.validation_issues:
                        logger.warning(
                            f"  Validation issues: {', '.join(speakers_result.validation_issues)}"
                        )
            else:
                logger.error(
                    f"✗ Step 2/3 FAILED: Speakers extraction returned invalid result type: {type(speakers_result)}"
                )
        except Exception as e:
            logger.error(
                f"✗ Step 2/3 FAILED: Speakers extraction raised exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise  # Re-raise to stop pipeline if speakers extraction fails

        # ========================================
        # Call 3: Duration Extraction
        # ========================================
        logger.info("Step 3/3: Extracting duration")
        duration_result = None
        try:
            duration_prompt = pipeline._get_user_prompt(
                "transcript_duration_extraction", prompt_context
            )
            duration_system_instruction = pipeline._get_system_instruction(
                "transcript_duration_extraction", prompt_context
            )

            # Skip approval on individual calls - will check on cumulative result
            duration_result = await pipeline._call_function(
                prompt=duration_prompt,
                system_instruction=duration_system_instruction,
                response_model=TranscriptDurationExtractionResult,
                step_name="transcript_duration_extraction",  # Internal step name, approval skipped
                step_number=execution_step_number,
                context=context,
                job_id=None,  # Skip approval check for individual calls
            )

            # Log success/failure for duration extraction
            if isinstance(duration_result, TranscriptDurationExtractionResult):
                confidence_str = (
                    f"{duration_result.confidence_score:.2f}"
                    if duration_result.confidence_score is not None
                    else "N/A"
                )
                if duration_result.duration_validated:
                    duration_min = (
                        f"{duration_result.duration/60:.1f}"
                        if duration_result.duration
                        else "N/A"
                    )
                    logger.info(
                        f"✓ Step 3/3 SUCCESS: Duration extraction completed - "
                        f"duration={duration_result.duration}s ({duration_min}min), "
                        f"method={duration_result.extraction_method or 'N/A'}, "
                        f"validated={duration_result.duration_validated}, "
                        f"confidence={confidence_str}"
                    )
                else:
                    logger.warning(
                        f"⚠ Step 3/3 PARTIAL: Duration extraction completed but validation failed - "
                        f"duration={duration_result.duration or 'N/A'}s, "
                        f"method={duration_result.extraction_method or 'N/A'}, "
                        f"validation_issues={len(duration_result.validation_issues)}, "
                        f"confidence={confidence_str}"
                    )
                    if duration_result.validation_issues:
                        logger.warning(
                            f"  Validation issues: {', '.join(duration_result.validation_issues)}"
                        )
            else:
                logger.error(
                    f"✗ Step 3/3 FAILED: Duration extraction returned invalid result type: {type(duration_result)}"
                )
        except Exception as e:
            logger.error(
                f"✗ Step 3/3 FAILED: Duration extraction raised exception: {type(e).__name__}: {str(e)}",
                exc_info=True,
            )
            raise  # Re-raise to stop pipeline if duration extraction fails

        # ========================================
        # Merge Results into TranscriptPreprocessingApprovalResult
        # ========================================
        logger.info("Merging results from 3 LLM calls")

        # Log summary of all three steps
        steps_summary = []
        if isinstance(content_result, TranscriptContentExtractionResult):
            status = "✓" if content_result.content_validated else "⚠"
            steps_summary.append(f"{status} Content")
        else:
            steps_summary.append("✗ Content")

        if isinstance(speakers_result, TranscriptSpeakersExtractionResult):
            status = "✓" if speakers_result.speakers_validated else "⚠"
            steps_summary.append(f"{status} Speakers")
        else:
            steps_summary.append("✗ Speakers")

        if isinstance(duration_result, TranscriptDurationExtractionResult):
            status = "✓" if duration_result.duration_validated else "⚠"
            steps_summary.append(f"{status} Duration")
        else:
            steps_summary.append("✗ Duration")

        logger.info(f"Transcript preprocessing summary: {' | '.join(steps_summary)}")

        # Aggregate validation issues
        all_validation_issues = []
        if isinstance(content_result, TranscriptContentExtractionResult):
            all_validation_issues.extend(content_result.validation_issues)
        if isinstance(speakers_result, TranscriptSpeakersExtractionResult):
            all_validation_issues.extend(speakers_result.validation_issues)
        if isinstance(duration_result, TranscriptDurationExtractionResult):
            all_validation_issues.extend(duration_result.validation_issues)

        # Determine overall validation status
        content_validated = (
            isinstance(content_result, TranscriptContentExtractionResult)
            and content_result.content_validated
        )
        speakers_validated = (
            isinstance(speakers_result, TranscriptSpeakersExtractionResult)
            and speakers_result.speakers_validated
        )
        duration_validated = (
            isinstance(duration_result, TranscriptDurationExtractionResult)
            and duration_result.duration_validated
        )
        transcript_type_validated = (
            isinstance(content_result, TranscriptContentExtractionResult)
            and content_result.transcript_type is not None
        )

        is_valid = content_validated and speakers_validated and duration_validated

        # Calculate average confidence score
        confidence_scores = []
        if (
            isinstance(content_result, TranscriptContentExtractionResult)
            and content_result.confidence_score is not None
        ):
            confidence_scores.append(content_result.confidence_score)
        if (
            isinstance(speakers_result, TranscriptSpeakersExtractionResult)
            and speakers_result.confidence_score is not None
        ):
            confidence_scores.append(speakers_result.confidence_score)
        if (
            isinstance(duration_result, TranscriptDurationExtractionResult)
            and duration_result.confidence_score is not None
        ):
            confidence_scores.append(duration_result.confidence_score)

        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores)
            if confidence_scores
            else 0.5
        )

        # Determine if approval is required
        requires_approval = (
            not is_valid or avg_confidence < 0.7 or len(all_validation_issues) > 0
        )

        # Build approval suggestions
        approval_suggestions = []
        if (
            isinstance(content_result, TranscriptContentExtractionResult)
            and content_result.extracted_content
        ):
            approval_suggestions.append("Content extracted and validated")
        if (
            isinstance(speakers_result, TranscriptSpeakersExtractionResult)
            and speakers_result.speakers
        ):
            approval_suggestions.append(
                f"Speakers extracted: {', '.join(speakers_result.speakers)}"
            )
        if (
            isinstance(duration_result, TranscriptDurationExtractionResult)
            and duration_result.duration
        ):
            approval_suggestions.append(
                f"Duration extracted: {duration_result.duration} seconds ({duration_result.extraction_method})"
            )

        # Create merged result
        result = TranscriptPreprocessingApprovalResult(
            is_valid=is_valid,
            speakers_validated=speakers_validated,
            duration_validated=duration_validated,
            content_validated=content_validated,
            transcript_type_validated=transcript_type_validated,
            validation_issues=all_validation_issues,
            speakers=(
                speakers_result.speakers
                if isinstance(speakers_result, TranscriptSpeakersExtractionResult)
                else []
            ),
            duration=(
                duration_result.duration
                if isinstance(duration_result, TranscriptDurationExtractionResult)
                else None
            ),
            transcript_type=(
                content_result.transcript_type
                if isinstance(content_result, TranscriptContentExtractionResult)
                else None
            ),
            content_summary=(
                content_result.content_summary
                if isinstance(content_result, TranscriptContentExtractionResult)
                else None
            ),
            confidence_score=avg_confidence,
            requires_approval=requires_approval,
            approval_suggestions=approval_suggestions,
            # Include quality metrics from content extraction
            quality_metrics=(
                content_result.quality_metrics
                if isinstance(content_result, TranscriptContentExtractionResult)
                else None
            ),
            # Preserve parsing information from input if available
            parsing_confidence=prompt_context.get("parsing_confidence"),
            detected_format=prompt_context.get("detected_format"),
            parsing_warnings=prompt_context.get("parsing_warnings", []),
            speaking_time_per_speaker=prompt_context.get("speaking_time_per_speaker"),
            detected_language=prompt_context.get("detected_language"),
            key_topics=prompt_context.get("key_topics", []),
            conversation_flow=prompt_context.get("conversation_flow"),
        )

        # ========================================
        # Check Approval on Cumulative Result
        # ========================================
        if job_id:
            try:
                from marketing_project.processors.approval_helper import (
                    ApprovalRequiredException,
                    check_and_create_approval_request,
                )
                from marketing_project.services.job_manager import (
                    JobStatus,
                    get_job_manager,
                )

                logger.info(
                    f"[APPROVAL] Checking approval for cumulative transcript preprocessing result in job {job_id}"
                )

                # Convert result to dict for approval system
                try:
                    result_dict = result.model_dump(mode="json")
                except (TypeError, ValueError):
                    result_dict = result.model_dump()

                # Extract confidence score
                confidence = result_dict.get("confidence_score")

                # Prepare input data for approval context
                pipeline_content = context.get("input_content") if context else None
                content_for_approval = (
                    pipeline_content
                    if pipeline_content
                    else {"title": "N/A", "content": "N/A"}
                )
                input_data = {
                    "prompt": "Transcript preprocessing (content, speakers, duration extraction)",
                    "system_instruction": "Cumulative result from 3 LLM extraction calls",
                    "context_keys": list(context.keys()) if context else [],
                    "original_content": pipeline_content or content_for_approval,
                }

                # Check if approval is needed (raises ApprovalRequiredException if required)
                await check_and_create_approval_request(
                    job_id=job_id,
                    agent_name="transcript_preprocessing_approval",
                    step_name=f"Step {execution_step_number}: transcript_preprocessing_approval (cumulative)",
                    step_number=execution_step_number,
                    input_data=input_data,
                    output_data=result_dict,
                    context=context or {},
                    confidence_score=confidence,
                    suggestions=result.approval_suggestions
                    or [
                        "Review transcript preprocessing results",
                        "Verify extracted content, speakers, and duration",
                        "Check validation issues if any",
                    ],
                )
                # If no exception raised, approval not needed or auto-approved, continue
                logger.info(
                    f"[APPROVAL] No approval required for cumulative result in job {job_id}, continuing pipeline"
                )

            except ApprovalRequiredException as e:
                # Approval required - pipeline should stop
                logger.info(
                    f"[APPROVAL] Cumulative transcript preprocessing result requires approval. "
                    f"Pipeline stopping. Approval ID: {e.approval_id}"
                )

                # Update job status to WAITING_FOR_APPROVAL
                job_manager = get_job_manager()
                await job_manager.update_job_status(
                    job_id, JobStatus.WAITING_FOR_APPROVAL
                )
                await job_manager.update_job_progress(
                    job_id,
                    90,
                    f"Waiting for approval at step {execution_step_number}: transcript_preprocessing_approval",
                )

                # Re-raise to stop pipeline
                raise

        # Merge AI-extracted data back into input_content for subsequent steps
        # This allows the pipeline to use extracted speakers, duration, etc.
        input_content = context.get("input_content", {})
        if isinstance(input_content, dict):
            # Update content with extracted content if available
            if (
                isinstance(content_result, TranscriptContentExtractionResult)
                and content_result.extracted_content
            ):
                input_content["content"] = content_result.extracted_content
                logger.info("Updated content with extracted content")

            # Update speakers if AI extracted them and they're missing from input
            if result.speakers and (
                not input_content.get("speakers")
                or len(input_content.get("speakers", [])) == 0
            ):
                input_content["speakers"] = result.speakers
                logger.info(
                    f"Merged AI-extracted speakers into input_content: {result.speakers}"
                )

            # Update duration if AI extracted it and it's missing from input
            if result.duration is not None and input_content.get("duration") is None:
                input_content["duration"] = result.duration
                logger.info(
                    f"Merged AI-extracted duration into input_content: {result.duration} seconds"
                )

            # Update transcript_type if AI validated/confirmed it
            if result.transcript_type and (
                not input_content.get("transcript_type")
                or input_content.get("transcript_type") == "podcast"
            ):
                input_content["transcript_type"] = result.transcript_type
                logger.info(
                    f"Updated transcript_type in input_content: {result.transcript_type}"
                )

            # Merge parsing information from result back to input_content
            if result.parsing_confidence is not None:
                input_content["parsing_confidence"] = result.parsing_confidence
            if result.detected_format:
                input_content["detected_format"] = result.detected_format
            if result.parsing_warnings:
                input_content["parsing_warnings"] = result.parsing_warnings
            if result.quality_metrics:
                input_content["quality_metrics"] = result.quality_metrics
            if result.speaking_time_per_speaker:
                input_content["speaking_time_per_speaker"] = (
                    result.speaking_time_per_speaker
                )
            if result.detected_language:
                input_content["detected_language"] = result.detected_language
            if result.key_topics:
                input_content["key_topics"] = result.key_topics
            if result.conversation_flow:
                input_content["conversation_flow"] = result.conversation_flow

            # Update context with modified input_content
            context["input_content"] = input_content

            # Log if AI successfully auto-fixed issues
            if (
                result.speakers_validated
                and result.duration_validated
                and result.content_validated
                and not result.requires_approval
            ):
                logger.info(
                    "AI successfully extracted missing data from 3 LLM calls - approval not required"
                )

        return result
