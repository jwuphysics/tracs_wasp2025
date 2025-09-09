"""Data models for multi-telescope classification analysis."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class TelescopeDetectionModel(BaseModel):
    """Model for detecting which telescopes are mentioned in a paper."""
    telescopes: List[str] = Field(..., description="List of telescopes detected in the paper from: CHANDRA, HST, JWST, or empty list for NONE")
    reasoning: str = Field(..., description="Explanation of which telescopes were detected and why")


class TelescopeClassificationModel(BaseModel):
    """Model for classifying a paper's relationship to a specific telescope."""
    telescope: str = Field(..., description="The telescope being classified (CHANDRA, HST, JWST, or NONE)")
    science: bool = Field(..., description="True if paper uses telescope data to obtain new scientific results")
    instrumentation: bool = Field(..., description="True if paper describes technical aspects of the telescope, calibration, or instrumentation")
    mention: bool = Field(..., description="True if paper references telescope but does not produce new results or contribute to understanding it")
    not_telescope: bool = Field(..., description="True if paper includes references confused with telescope but is actually about something else")
    quotes: List[str] = Field(..., description="Supporting quotes from the text (exact substrings)")
    reasoning: str = Field(..., description="Detailed explanation of the classification decision")


class MultiTelescopeResult(BaseModel):
    """Complete result for a paper analyzing all telescopes."""
    bibcode: str = Field(..., description="Paper identifier")
    detected_telescopes: List[str] = Field(..., description="All telescopes detected in the paper")
    classifications: List[TelescopeClassificationModel] = Field(..., description="Classification results for each telescope")
    primary_telescope: str = Field(..., description="Primary telescope for competition output (CHANDRA, HST, JWST, or NONE)")
    
    
class CSVInputRow(BaseModel):
    """Model for CSV input row with combined Id field."""
    id: str = Field(..., description="Combined ID (bibcode_telescope)")
    bibcode: str = Field(..., description="Extracted bibcode")
    telescope: str = Field(..., description="Specified telescope for this row")
    author: str = Field(default="", description="Paper author(s)")
    year: Optional[int] = Field(default=None, description="Publication year")
    title: str = Field(default="", description="Paper title")
    abstract: str = Field(default="", description="Paper abstract")
    body: str = Field(default="", description="Paper body text")
    acknowledgments: str = Field(default="", description="Paper acknowledgments")
    grants: str = Field(default="", description="Grant information")


class SingleTelescopeResult(BaseModel):
    """Result for classifying a specific telescope for a paper (CSV mode)."""
    id: str = Field(..., description="Combined ID (bibcode_telescope)")
    bibcode: str = Field(..., description="Paper identifier")
    telescope: str = Field(..., description="Telescope being classified")
    classification: TelescopeClassificationModel = Field(..., description="Classification result for this telescope")


class CompetitionOutput(BaseModel):
    """Model for competition CSV output format."""
    Id: str = Field(..., description="Paper identifier (combined bibcode_telescope)")
    telescope: str = Field(..., description="Primary telescope (CHANDRA, HST, JWST, or NONE)")
    science: bool = Field(..., description="Science classification for primary telescope")
    instrumentation: bool = Field(..., description="Instrumentation classification for primary telescope")
    mention: bool = Field(..., description="Mention classification for primary telescope")
    not_telescope: bool = Field(..., description="Not telescope classification for primary telescope")


# Legacy models kept for reference but unused in new system
class TelescopeIdentificationModel(BaseModel):
    """Legacy model - replaced by TelescopeDetectionModel."""
    telescope: str = Field(..., description="Primary telescope discussed in the paper (CHANDRA, HST, JWST, or NONE)")
    reason: str = Field(..., description="Justification for telescope identification based on the content")


class TelescopeClassificationReasoningModel(BaseModel):
    """Legacy model - functionality integrated into TelescopeClassificationModel."""
    quotes: list[str] = Field(..., description="A list of quotes supporting the analysis, MUST be exact substrings from the provided excerpts.")
    reason: str = Field(..., description="Complete analysis of how this paper relates to the identified telescope based ONLY on the provided excerpts")


class TelescopeClassificationScoringModel(BaseModel):
    """Legacy model - functionality integrated into TelescopeClassificationModel."""
    science: bool = Field(..., description="True if paper uses telescope data to obtain new scientific results")
    instrumentation: bool = Field(..., description="True if paper describes technical aspects of the telescope, calibration, or instrumentation")
    mention: bool = Field(..., description="True if paper references telescope but does not produce new results or contribute to understanding it")
    not_telescope: bool = Field(..., description="True if paper includes references confused with telescope but is actually about something else")