"""Pydantic models for structured treatment responses."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class Diagnosis(BaseModel):
    """Diagnosis information."""
    disease_name: str = Field(..., description="Name of the disease/problem identified")
    symptoms: List[str] = Field(..., description="List of observed symptoms")
    severity: str = Field(..., description="Severity level: mild/moderate/severe")
    affected_parts: List[str] = Field(..., description="Plant parts affected")


class ImmediateTreatment(BaseModel):
    """Immediate treatment actions."""
    actions: List[str] = Field(..., description="Immediate actions to take")
    emergency_measures: List[str] = Field(..., description="Emergency measures if applicable")
    timeline: str = Field(..., description="When to implement: immediate/within 24 hours/within 3 days")


class WeeklyTreatmentStep(BaseModel):
    """Treatment step for a specific week."""
    actions: List[str] = Field(..., description="Actions to take this week")
    monitoring: str = Field(..., description="What to monitor during this week")
    expected_results: str = Field(..., description="What results to expect")


class WeeklyTreatmentPlan(BaseModel):
    """Weekly treatment plan."""
    week_1: WeeklyTreatmentStep = Field(..., description="Week 1 treatment plan")
    week_2: WeeklyTreatmentStep = Field(..., description="Week 2 treatment plan")
    week_3: WeeklyTreatmentStep = Field(..., description="Week 3 treatment plan")
    week_4: WeeklyTreatmentStep = Field(..., description="Week 4 treatment plan")


class MedicineDetails(BaseModel):
    """Details for a specific medicine."""
    medicine_name: str = Field(..., description="Name of the medicine/chemical")
    active_ingredient: str = Field(..., description="Active ingredient name")
    dosage: str = Field(..., description="Concentration and quantity per application")
    application_method: str = Field(..., description="How to apply the medicine")
    frequency: str = Field(..., description="How often to apply")
    duration: Optional[str] = Field(None, description="Total treatment period")
    precautions: Optional[List[str]] = Field(None, description="Safety precautions")
    when_to_use: Optional[str] = Field(None, description="Conditions when to use this medicine")


class OrganicAlternative(BaseModel):
    """Organic treatment alternative."""
    name: str = Field(..., description="Name of the organic treatment")
    preparation: str = Field(..., description="How to prepare the treatment")
    application: str = Field(..., description="How to apply the treatment")


class MedicineRecommendations(BaseModel):
    """Medicine recommendations with primary and secondary treatments."""
    primary_treatment: MedicineDetails = Field(..., description="Primary medicine recommendation")
    secondary_treatment: Optional[MedicineDetails] = Field(None, description="Secondary/supportive medicine")
    organic_alternatives: List[OrganicAlternative] = Field(default=[], description="Organic treatment options")


class Prevention(BaseModel):
    """Prevention strategies."""
    cultural_practices: List[str] = Field(..., description="Cultural practices for prevention")
    crop_management: List[str] = Field(..., description="Crop management strategies")
    environmental_controls: List[str] = Field(..., description="Environmental control measures")
    monitoring_schedule: str = Field(..., description="When and what to monitor")


class AdditionalNotes(BaseModel):
    """Additional notes and considerations."""
    weather_considerations: str = Field(..., description="Weather-related advice")
    crop_stage_specific: str = Field(..., description="Stage-specific recommendations")
    regional_considerations: str = Field(..., description="Regional/location specific advice")
    follow_up: str = Field(..., description="When to seek further help")


class StructuredTreatmentResponse(BaseModel):
    """Complete structured treatment response."""
    diagnosis: Diagnosis = Field(..., description="Disease diagnosis information")
    immediate_treatment: ImmediateTreatment = Field(..., description="Immediate treatment actions")
    weekly_treatment_plan: WeeklyTreatmentPlan = Field(..., description="4-week treatment plan")
    medicine_recommendations: MedicineRecommendations = Field(..., description="Medicine recommendations")
    prevention: Prevention = Field(..., description="Prevention strategies")
    additional_notes: AdditionalNotes = Field(..., description="Additional notes and considerations")


class StructuredTreatmentQueryResponse(BaseModel):
    """API response model for structured treatment queries."""
    treatment: StructuredTreatmentResponse = Field(..., description="Structured treatment response")
    collection_used: str = Field(..., description="ChromaDB collection used")
    query_time: Optional[float] = Field(None, description="Query execution time in seconds")
    success: bool = Field(..., description="Whether the query was successful")
    raw_response: Optional[str] = Field(None, description="Raw LLM response for debugging")
    parsing_success: bool = Field(..., description="Whether JSON parsing was successful")
