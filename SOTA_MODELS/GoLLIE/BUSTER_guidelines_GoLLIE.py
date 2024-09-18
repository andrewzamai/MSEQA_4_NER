__package__ = "SOTA_MODELS.GoLLIE"
from typing import List
from GoLLIE.src.tasks.utils_typing import Entity, dataclass

"""
BUSTER entity definitions
"""

@dataclass
class GenericConsultingCompany(Entity):
    """ Refers to a business entity that provides non-legal advisory services in areas such as finance, accounting, due diligence, and other professional consulting services.
    Avoid labeling a company that primarily provides legal services. Exercise caution with company names that include personal names which might be confused with individuals,
    and consider the context to determine whether the reference is to a company.
    """

    span: str  # Such as: "Stifel Nicolaus Weisel", "Daroth Capital Advisors", "BofA Merrill Lynch"


@dataclass
class LegalConsultingCompany(Entity):
    """Refers to a business entity providing legal services and advice on matters such as government regulation, litigation, anti-trust, structured finance, and tax, among others.
    Exercise caution when labeling entities that might also refer to individuals or groups of people, such as 'Morgan Securities' (could refer to a company or a person).
    Take care to distinguish between names that may appear to be company names but are in fact personal names.
    """

    span: str  # Such as: "Hogan Lovells US LLP", "Kirkland & Ellis LLP", "Morrison & Foerster LLP"


@dataclass
class AnnualRevenues(Entity):
    """Refers to the past or present total income earned by a company or entity within a specific fiscal year.
    Do not annotate generic terms such as 'revenue', but the numerical amounts of revenue, for example '120 million'. Do not annotate half-yearly revenues and future revenue forecasts.
    """

    span: str  # Such as: "US$ 900 million", "US$ 1 billion", "150 billion dollars"


@dataclass
class AcquiredCompany(Entity):
    """Refers to a company that is beeing acquired by another company through a business transaction, such as a merger or acquisition.
    Ensure that the labeled entity is indeed a company and not a product. The company's role as 'being acquired' must be understandable from the sentence in which it appears. Do not label company names which role in the transaction is not clear.
    """

    span: str  # Such as: "Liberty Media Corporation", "Security Holding Corp. (SHC)", "Somaxon Pharmaceuticals , Inc ."


@dataclass
class BuyingCompany(Entity):
    """Refers to a company that is buying another company or its assets through a transaction or merger.
    When recognizing 'buying company' entities, focus on the company names directly involved in the acquisition process as buyers, while being careful not to label subsidiaries or companies in other roles. The company's role as 'buyer' must be understandable from the sentence in which it appears. Do not label company names which role in the transaction is not evident.
    """

    span: str  # Such as: "Liberty Media Corporation", "Security Holding Corp. (SHC)", "Somaxon Pharmaceuticals , Inc ."


@dataclass
class SellingCompany(Entity):
    """Refers to a company that is selling or divesting assets, subsidiaries, or equity to another party as part of a transaction.
    Be careful when identifying the entity actually doing the selling, as it may not be the main subject of the sentence or document. Pay attention to possessive forms and synonyms such as 'vendor', 'owner', or 'parent company'. The company's role as 'seller' must be understandable from the sentence in which it appears. Do not label company names which role in the transaction is not evident.
    """

    span: str  # Such as: "Liberty Media Corporation", "Security Holding Corp. (SHC)", "Somaxon Pharmaceuticals , Inc ."


ENTITY_DEFINITIONS: List = [
    GenericConsultingCompany,
    LegalConsultingCompany,
    AnnualRevenues,
    AcquiredCompany,
    BuyingCompany,
    SellingCompany
]