"""ORCID API integration tools for researcher profile retrieval."""
import logging
import os
from typing import Dict, Optional, Any, List
from datetime import datetime
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from .base import BaseToolError

load_dotenv()

logger = logging.getLogger(__name__)


class OrcidError(BaseToolError):
    """Custom exception for ORCID API errors."""


@dataclass
class ResearcherProfile:
    """Simplified researcher profile for Gemini integration."""
    orcid_id: str
    name: str
    biography: Optional[str] = None
    keywords: List[str] = None
    emails: List[str] = None
    affiliations: List[Dict[str, Any]] = None
    publications: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.emails is None:
            self.emails = []
        if self.affiliations is None:
            self.affiliations = []
        if self.publications is None:
            self.publications = []


class OrcidConfig(BaseModel):
    """Configuration for ORCID API client."""
    api_key: Optional[str] = None
    timeout: int = Field(default=30, ge=1)
    base_url: str = Field(default="https://pub.orcid.org/v3.0")
    
    @property
    def headers(self) -> Dict[str, str]:
        headers = {'Accept': 'application/vnd.orcid+json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        return headers


class OrcidTools:
    """Tools for ORCID API integration."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize ORCID client.
        
        Args:
            api_key: ORCID API key. Optional for public data access.
        """
        self.config = OrcidConfig(api_key=api_key)

    async def _fetch_orcid_data(self, orcid_id: str, endpoint: str) -> Dict[str, Any]:
        """Generic function to fetch data from ORCID API."""
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            try:
                response = await client.get(
                    f"{self.config.base_url}/{orcid_id}/{endpoint}",
                    headers=self.config.headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    raise OrcidError(f"ORCID ID not found: {orcid_id}") from e
                raise OrcidError(f"Failed to fetch {endpoint} data: {str(e)}") from e
            except httpx.RequestError as e:
                raise OrcidError(f"Network error fetching {endpoint}: {str(e)}") from e

    def _extract_name(self, person_data: Dict[str, Any]) -> str:
        """Extract researcher name from ORCID person data."""
        name_data = person_data.get('name', {})
        
        # Try credit name first, then given + family names
        credit_name = name_data.get('credit-name', {})
        if credit_name and credit_name.get('value'):
            return credit_name['value']
        
        given_names = name_data.get('given-names', {})
        family_name = name_data.get('family-name', {})
        
        given = given_names.get('value', '') if given_names else ''
        family = family_name.get('value', '') if family_name else ''
        
        return f"{given} {family}".strip() or "Unknown Name"

    def _extract_biography(self, person_data: Dict[str, Any]) -> Optional[str]:
        """Extract biography from ORCID person data."""
        biography = person_data.get('biography')
        if biography and biography.get('content'):
            return biography['content']
        return None

    def _extract_keywords(self, person_data: Dict[str, Any]) -> List[str]:
        """Extract keywords from ORCID person data."""
        keywords_data = person_data.get('keywords', {})
        keywords = keywords_data.get('keyword', [])
        if not isinstance(keywords, list):
            keywords = [keywords] if keywords else []
        
        return [kw.get('content', '') for kw in keywords if kw.get('content')]

    def _extract_emails(self, person_data: Dict[str, Any]) -> List[str]:
        """Extract email addresses from ORCID person data."""
        emails_data = person_data.get('emails', {})
        emails = emails_data.get('email', [])
        if not isinstance(emails, list):
            emails = [emails] if emails else []
        
        return [email.get('email', '') for email in emails if email.get('email')]

    def _extract_affiliations(self, education_data: Dict[str, Any], employment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract affiliations from ORCID education and employment data."""
        affiliations = []
        
        # Process education
        for group in education_data.get('affiliation-group', []):
            for summary in group.get('summaries', []):
                edu_summary = summary.get('education-summary')
                if edu_summary:
                    org = edu_summary.get('organization', {})
                    affiliation = {
                        'type': 'education',
                        'institution': org.get('name', 'Unknown Institution'),
                        'department': edu_summary.get('department-name'),
                        'role': edu_summary.get('role-title'),
                        'start_date': self._format_date(edu_summary.get('start-date')),
                        'end_date': self._format_date(edu_summary.get('end-date'))
                    }
                    affiliations.append(affiliation)
        
        # Process employment
        for group in employment_data.get('affiliation-group', []):
            for summary in group.get('summaries', []):
                emp_summary = summary.get('employment-summary')
                if emp_summary:
                    org = emp_summary.get('organization', {})
                    affiliation = {
                        'type': 'employment',
                        'institution': org.get('name', 'Unknown Institution'),
                        'department': emp_summary.get('department-name'),
                        'role': emp_summary.get('role-title'),
                        'start_date': self._format_date(emp_summary.get('start-date')),
                        'end_date': self._format_date(emp_summary.get('end-date'))
                    }
                    affiliations.append(affiliation)
        
        return affiliations
    
    def _extract_publications(self, works_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract publications from ORCID works data."""
        publications = []
        
        for group in works_data.get('group', []):
            for summary in group.get('work-summary', []):
                # Get basic publication info
                title_data = summary.get('title', {})
                title = title_data.get('title', {}).get('value') if title_data else None
                
                if not title:
                    continue
                
                publication = {
                    'title': title,
                    'type': summary.get('type'),
                    'journal': summary.get('journal-title', {}).get('value') if summary.get('journal-title') else None,
                    'publication_date': self._format_date(summary.get('publication-date')),
                    'url': summary.get('url', {}).get('value') if summary.get('url') else None
                }
                
                # Extract DOI from external IDs
                external_ids = group.get('external-ids', {}).get('external-id', [])
                if not isinstance(external_ids, list):
                    external_ids = [external_ids] if external_ids else []
                
                for ext_id in external_ids:
                    if ext_id.get('external-id-type') == 'doi':
                        publication['doi'] = ext_id.get('external-id-value')
                        break
                
                publications.append(publication)
        
        return publications

    def _format_date(self, date_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Format ORCID date data to readable string."""
        if not date_data:
            return None
        
        year = date_data.get('year', {}).get('value') if date_data.get('year') else None
        month = date_data.get('month', {}).get('value') if date_data.get('month') else None
        day = date_data.get('day', {}).get('value') if date_data.get('day') else None
        
        if year:
            date_parts = [str(year)]
            if month:
                date_parts.append(f"{int(month):02d}")
                if day:
                    date_parts.append(f"{int(day):02d}")
            return "-".join(date_parts)
        
        return None

    async def get_researcher_profile(self, orcid_id: str) -> ResearcherProfile:
        """Fetch comprehensive researcher profile from ORCID."""
        try:
            # Fetch all required data
            person_data = await self._fetch_orcid_data(orcid_id, "person")
            works_data = await self._fetch_orcid_data(orcid_id, "works")
            education_data = await self._fetch_orcid_data(orcid_id, "educations")
            employment_data = await self._fetch_orcid_data(orcid_id, "employments")
            
            # Extract information using existing parsing logic
            name = self._extract_name(person_data)
            biography = self._extract_biography(person_data)
            keywords = self._extract_keywords(person_data)
            emails = self._extract_emails(person_data)
            affiliations = self._extract_affiliations(education_data, employment_data)
            publications = self._extract_publications(works_data)
            
            return ResearcherProfile(
                orcid_id=orcid_id,
                name=name,
                biography=biography,
                keywords=keywords,
                emails=emails,
                affiliations=affiliations,
                publications=publications
            )
            
        except Exception as e:
            if not isinstance(e, OrcidError):
                logger.error("Error processing researcher profile for ORCID %s: %s", orcid_id, str(e))
                raise OrcidError(f"Failed to process researcher profile: {str(e)}") from e
            raise


# Global instance for easy access  
_ORCID_TOOLS = None


def get_orcid_tools() -> OrcidTools:
    """Get singleton instance of OrcidTools."""
    global _ORCID_TOOLS
    if _ORCID_TOOLS is None:
        _ORCID_TOOLS = OrcidTools()
    return _ORCID_TOOLS


# Tool functions for Gemini integration
async def get_researcher_profile(orcid_id: str) -> Dict[str, Any]:
    """Get comprehensive researcher profile from ORCID.

    Args:
        orcid_id: ORCID identifier (e.g., "0000-0000-0000-0000")

    Returns:
        Dictionary containing researcher profile information including name, 
        biography, keywords, affiliations, and publications
    """
    logger.info(f"👤 Fetching researcher profile for ORCID: {orcid_id}")
    
    try:
        profile = await get_orcid_tools().get_researcher_profile(orcid_id)
        
        result = {
            "orcid_id": profile.orcid_id,
            "name": profile.name,
            "biography": profile.biography,
            "keywords": profile.keywords,
            "emails": profile.emails,
            "affiliations": profile.affiliations,
            "publications": profile.publications,
            "publication_count": len(profile.publications),
            "affiliation_count": len(profile.affiliations)
        }
        
        logger.info(f"✅ Profile retrieved: {profile.name} ({len(profile.publications)} publications)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Failed to get researcher profile: {str(e)[:200]}")
        raise