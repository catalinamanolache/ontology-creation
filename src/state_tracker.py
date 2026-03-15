import json
from src.schemas import BootstrapResult, OntologyClass, OntologyProperty

class KnowledgeStateTracker:
    def __init__(self):
        self.approved_classes = {}
        self.approved_properties = {}

    def ingest_bootstrap(self, bootstrap_result: BootstrapResult):
        """
        Takes the initial Phase 3 schema and locks it into the global state.
        """
        for cls in bootstrap_result.classes:
            self.approved_classes[cls.uri] = cls.comment
            
        for prop in bootstrap_result.properties:
            self.approved_properties[prop.uri] = {
                "domain": prop.domain,
                "range": prop.range,
                "comment": prop.comment
            }

    def format_for_prompt(self) -> str:
        """
        Formats the current approved ontology state into a string for LLM injection.
        """
        state_dict = {
            "Classes": self.approved_classes,
            "Properties": self.approved_properties
        }
        return json.dumps(state_dict, indent=2)

    def get_ontology_size(self) -> int:
        """
        Returns the total number of classes and properties to measure Delta O.
        """
        return len(self.approved_classes) + len(self.approved_properties)
