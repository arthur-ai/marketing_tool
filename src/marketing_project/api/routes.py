"""
Centralized API Routes Registry

This file provides a comprehensive view of all API endpoints in the system.
"""

from fastapi import APIRouter

# Import all sub-routers directly from their modules to avoid circular imports
from marketing_project.api.admin_users import router as admin_users_router
from marketing_project.api.analytics import router as analytics_router
from marketing_project.api.approvals import router as approvals_router
from marketing_project.api.batch import router as batch_router
from marketing_project.api.content import router as content_router
from marketing_project.api.core import router as core_router
from marketing_project.api.design_kit import router as design_kit_router
from marketing_project.api.feedback import router as feedback_router
from marketing_project.api.health import router as health_router
from marketing_project.api.internal_docs import router as internal_docs_router
from marketing_project.api.jobs import router as jobs_router
from marketing_project.api.processors import router as processors_router
from marketing_project.api.settings import router as settings_router
from marketing_project.api.social_media import router as social_media_router
from marketing_project.api.step_results import router as step_results_router
from marketing_project.api.system import router as system_router
from marketing_project.api.upload import router as upload_router
from marketing_project.api.user_settings import router as user_settings_router


def register_routes() -> APIRouter:
    """
    Register all API routes in one centralized location.

    Returns:
        APIRouter with all sub-routers included
    """
    # Create main API router
    api_router = APIRouter(prefix="/api/v1", tags=["Marketing API"])

    # ========================================
    # CORE PIPELINE ROUTES
    # ========================================
    # POST /api/v1/analyze - Analyze content
    # POST /api/v1/pipeline - Run complete pipeline
    api_router.include_router(core_router, tags=["Core Pipeline"])

    # ========================================
    # PROCESSOR ROUTES (Direct Processing)
    # ========================================
    # POST /api/v1/process/blog - Process blog posts
    # POST /api/v1/process/release-notes - Process release notes
    # POST /api/v1/process/transcript - Process transcripts
    api_router.include_router(processors_router, tags=["Processors"])

    # ========================================
    # JOB MANAGEMENT ROUTES
    # ========================================
    # GET /api/v1/jobs - List all jobs
    # GET /api/v1/jobs/{job_id} - Get job details
    # GET /api/v1/jobs/{job_id}/status - Get job status
    # GET /api/v1/jobs/{job_id}/result - Get job result
    # DELETE /api/v1/jobs/{job_id} - Delete job
    api_router.include_router(jobs_router, prefix="/jobs", tags=["Jobs"])

    # ========================================
    # APPROVAL ROUTES
    # ========================================
    # GET /api/v1/approvals/pending - Get pending approvals
    # GET /api/v1/approvals/{approval_id} - Get approval details
    # POST /api/v1/approvals/{approval_id}/approve - Approve
    # POST /api/v1/approvals/{approval_id}/reject - Reject
    # POST /api/v1/approvals/{approval_id}/modify - Modify and approve
    api_router.include_router(approvals_router, prefix="/approvals", tags=["Approvals"])

    # ========================================
    # STEP RESULTS ROUTES
    # ========================================
    # GET /api/v1/results/jobs - List all jobs with results
    # GET /api/v1/results/jobs/{job_id} - Get job step results
    # GET /api/v1/results/jobs/{job_id}/steps/{step_filename} - Get step content
    # GET /api/v1/results/jobs/{job_id}/steps/{step_filename}/download - Download step
    # DELETE /api/v1/results/jobs/{job_id} - Delete job results
    api_router.include_router(step_results_router, tags=["Step Results"])

    # ========================================
    # CONTENT SOURCE ROUTES
    # ========================================
    # GET /api/v1/content-sources - List content sources
    # GET /api/v1/content-sources/{source_name}/status - Get source status
    # POST /api/v1/content-sources/{source_name}/fetch - Fetch from source
    api_router.include_router(content_router, tags=["Content Sources"])

    # ========================================
    # HEALTH & SYSTEM ROUTES
    # ========================================
    # GET /api/v1/health - Health check
    # GET /api/v1/ready - Readiness check
    api_router.include_router(health_router, tags=["Health"])

    # GET /api/v1/config - Get system config
    # GET /api/v1/pipeline/spec - Get pipeline spec
    api_router.include_router(system_router, tags=["System"])

    # ========================================
    # FILE UPLOAD ROUTES
    # ========================================
    # POST /api/v1/upload - Upload file
    api_router.include_router(upload_router, tags=["File Upload"])

    # ========================================
    # ANALYTICS ROUTES
    # ========================================
    # GET /api/v1/analytics/dashboard - Get dashboard stats
    # GET /api/v1/analytics/pipeline - Get pipeline stats
    # GET /api/v1/analytics/content - Get content stats
    # GET /api/v1/analytics/recent-activity - Get recent activity
    # GET /api/v1/analytics/trends - Get trend data
    api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics"])

    # ========================================
    # INTERNAL DOCS CONFIGURATION ROUTES
    # ========================================
    # GET /api/v1/internal-docs/config - Get active config
    # GET /api/v1/internal-docs/config/{version} - Get config by version
    # POST /api/v1/internal-docs/config - Create/update config
    # GET /api/v1/internal-docs/versions - List versions
    # POST /api/v1/internal-docs/activate/{version} - Activate version
    api_router.include_router(
        internal_docs_router, prefix="/internal-docs", tags=["Internal Docs"]
    )

    # ========================================
    # DESIGN KIT CONFIGURATION ROUTES
    # ========================================
    # GET /api/v1/design-kit/config - Get active config
    # GET /api/v1/design-kit/config/{version} - Get config by version
    # GET /api/v1/design-kit/config/{content_type}/type - Get content-type-specific config
    # POST /api/v1/design-kit/config - Create/update config (manual)
    # POST /api/v1/design-kit/generate - Generate config from analysis
    # GET /api/v1/design-kit/versions - List versions
    # POST /api/v1/design-kit/activate/{version} - Activate version
    api_router.include_router(
        design_kit_router, prefix="/design-kit", tags=["Design Kit"]
    )

    # ========================================
    # SOCIAL MEDIA ROUTES
    # ========================================
    # POST /api/v1/social-media/preview - Preview post
    # POST /api/v1/social-media/validate - Validate post
    # POST /api/v1/social-media/update - Update post
    api_router.include_router(social_media_router, tags=["Social Media"])

    # ========================================
    # FEEDBACK ROUTES
    # ========================================
    # POST /api/v1/feedback - Submit feedback
    # GET /api/v1/feedback/stats - Get feedback statistics
    api_router.include_router(feedback_router, tags=["Feedback"])

    # ========================================
    # BATCH PROCESSING ROUTES
    # ========================================
    # POST /api/v1/batch/blog - Process multiple blog posts
    # GET /api/v1/batch/campaign/{campaign_id}/jobs - Get campaign jobs
    api_router.include_router(batch_router, tags=["Batch Processing"])

    # ========================================
    # SETTINGS ROUTES
    # ========================================
    # GET /api/v1/settings/pipeline - Get pipeline settings
    # POST /api/v1/settings/pipeline - Save pipeline settings
    api_router.include_router(settings_router, tags=["Settings"])

    # ========================================
    # ADMIN USER MANAGEMENT ROUTES
    # ========================================
    # GET /api/v1/admin/users                      - [Admin] List all users
    # GET /api/v1/admin/users/{user_id}             - [Admin] Get user by ID
    # GET /api/v1/admin/users/{user_id}/roles       - [Admin] Get user roles
    # PUT /api/v1/admin/users/{user_id}/roles       - [Admin] Set user roles
    api_router.include_router(
        admin_users_router, prefix="/admin/users", tags=["Admin - User Management"]
    )

    # ========================================
    # USER SETTINGS ROUTES
    # ========================================
    # GET /api/v1/users/me/settings - Get own settings
    # PUT /api/v1/users/me/settings - Upsert own settings
    # DELETE /api/v1/users/me/settings - Reset own settings
    # GET /api/v1/users/{user_id}/settings - [Admin] Get user settings
    # PUT /api/v1/users/{user_id}/settings - [Admin] Update user settings
    api_router.include_router(
        user_settings_router, prefix="/users", tags=["User Settings"]
    )

    return api_router


# ========================================
# ROUTE SUMMARY
# ========================================
"""
TOTAL ENDPOINTS: ~30+

Core Pipeline (2):
  - POST /api/v1/analyze
  - POST /api/v1/pipeline

Processors (3):
  - POST /api/v1/process/blog
  - POST /api/v1/process/release-notes
  - POST /api/v1/process/transcript

Jobs (5):
  - GET /api/v1/jobs
  - GET /api/v1/jobs/{job_id}
  - GET /api/v1/jobs/{job_id}/status
  - GET /api/v1/jobs/{job_id}/result
  - DELETE /api/v1/jobs/{job_id}

Approvals (5):
  - GET /api/v1/approvals/pending
  - GET /api/v1/approvals/{approval_id}
  - POST /api/v1/approvals/{approval_id}/approve
  - POST /api/v1/approvals/{approval_id}/reject
  - POST /api/v1/approvals/{approval_id}/modify

Step Results (5):
  - GET /api/v1/results/jobs
  - GET /api/v1/results/jobs/{job_id}
  - GET /api/v1/results/jobs/{job_id}/steps/{step_filename}
  - GET /api/v1/results/jobs/{job_id}/steps/{step_filename}/download
  - DELETE /api/v1/results/jobs/{job_id}

Content Sources (3):
  - GET /api/v1/content-sources
  - GET /api/v1/content-sources/{source_name}/status
  - POST /api/v1/content-sources/{source_name}/fetch

Health & System (4):
  - GET /api/v1/health
  - GET /api/v1/ready
  - GET /api/v1/config
  - GET /api/v1/pipeline/spec

File Upload (1):
  - POST /api/v1/upload

Analytics (5):
  - GET /api/v1/analytics/dashboard
  - GET /api/v1/analytics/pipeline
  - GET /api/v1/analytics/content
  - GET /api/v1/analytics/recent-activity
  - GET /api/v1/analytics/trends
"""
