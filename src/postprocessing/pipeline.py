"""
Post-processing pipeline.

Provides a modular way to compute derived fields from primary solver outputs.
Each processor is independent and can be added/removed without affecting others.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Set
from dataclasses import dataclass

from .fields import FieldData
from .fluid import FluidState


class PostProcessor(ABC):
    """
    Base class for post-processors.
    
    Each processor computes one or more fields from existing fields.
    Processors declare their requirements and outputs, allowing the
    pipeline to validate and order them correctly.
    
    Example:
        class PressureProcessor(PostProcessor):
            @property
            def requires(self) -> Set[str]:
                return {"velocity"}
            
            @property
            def produces(self) -> Set[str]:
                return {"pressure", "pressure_coefficient"}
            
            def process(self, fields: FieldData, fluid: FluidState) -> None:
                vel = fields.velocity
                # ... compute pressure ...
                fields.add_scalar("pressure", P, units="Pa")
    """
    
    @property
    @abstractmethod
    def requires(self) -> Set[str]:
        """Field names this processor needs as input."""
        pass
    
    @property
    @abstractmethod
    def produces(self) -> Set[str]:
        """Field names this processor will create."""
        pass
    
    @property
    def name(self) -> str:
        """Processor name (defaults to class name)."""
        return self.__class__.__name__
    
    @abstractmethod
    def process(self, fields: FieldData, fluid: Optional[FluidState] = None) -> None:
        """
        Compute fields and add them to the FieldData container.
        
        Args:
            fields: Field container with existing fields
            fluid: Optional fluid properties (required by some processors)
        """
        pass
    
    def validate(self, fields: FieldData) -> bool:
        """Check if required fields are available."""
        missing = self.requires - set(fields.available)
        if missing:
            raise ValueError(f"{self.name} requires fields: {missing}")
        return True


class ProcessorPipeline:
    """
    Manages a chain of post-processors.
    
    Automatically orders processors based on their dependencies.
    Validates that all requirements can be satisfied.
    
    Usage:
        pipeline = ProcessorPipeline()
        pipeline.add(VelocityMagnitudeProcessor())
        pipeline.add(PressureProcessor())
        pipeline.add(VelocityPotentialProcessor())
        
        # Run all processors
        pipeline.run(fields, fluid)
        
        # Or run specific ones
        pipeline.run(fields, fluid, only=["pressure"])
    """
    
    def __init__(self):
        self._processors: List[PostProcessor] = []
    
    def add(self, processor: PostProcessor) -> ProcessorPipeline:
        """Add a processor to the pipeline."""
        self._processors.append(processor)
        return self  # Allow chaining
    
    def clear(self) -> None:
        """Remove all processors."""
        self._processors.clear()
    
    @property
    def processors(self) -> List[PostProcessor]:
        """Get ordered list of processors."""
        return self._order_processors()
    
    def _order_processors(self) -> List[PostProcessor]:
        """
        Topologically sort processors based on dependencies.
        
        Returns processors in an order where each processor's
        requirements are satisfied by previous processors.
        """
        # Build dependency graph
        available: Set[str] = set()
        ordered: List[PostProcessor] = []
        remaining = list(self._processors)
        
        # Note: Initial available fields will be checked at runtime
        # Here we just order based on inter-processor dependencies
        
        max_iterations = len(remaining) * 2
        iteration = 0
        
        while remaining and iteration < max_iterations:
            iteration += 1
            for proc in remaining[:]:
                # Check if this processor's requirements are satisfied
                # by already-ordered processors (or will be available initially)
                if proc.requires <= available or not proc.requires:
                    ordered.append(proc)
                    available.update(proc.produces)
                    remaining.remove(proc)
        
        if remaining:
            # Some processors have unmet dependencies
            # Add them anyway - will fail at runtime with clear error
            ordered.extend(remaining)
        
        return ordered
    
    def run(self, 
            fields: FieldData, 
            fluid: Optional[FluidState] = None,
            only: Optional[List[str]] = None) -> FieldData:
        """
        Run the pipeline.
        
        Args:
            fields: Field container with initial fields (e.g., velocity)
            fluid: Fluid properties (required by some processors)
            only: If specified, only run processors that produce these fields
            
        Returns:
            The same FieldData object, now populated with computed fields
        """
        processors = self.processors
        
        for proc in processors:
            # Skip if we only want specific outputs
            if only is not None:
                if not proc.produces.intersection(only):
                    continue
            
            # Validate requirements
            proc.validate(fields)
            
            # Run processor
            proc.process(fields, fluid)
        
        return fields
    
    def available_outputs(self) -> Set[str]:
        """Get all field names this pipeline can produce."""
        outputs = set()
        for proc in self._processors:
            outputs.update(proc.produces)
        return outputs
    
    def __repr__(self) -> str:
        names = [p.name for p in self._processors]
        return f"ProcessorPipeline({names})"


# -----------------------------------------------------------------------------
# Default pipeline factory
# -----------------------------------------------------------------------------

def create_default_pipeline() -> ProcessorPipeline:
    """
    Create a pipeline with all standard processors.
    
    Includes:
    - PressureProcessor (velocity → pressure, Cp)
    - VelocityPotentialProcessor (velocity → phi)
    """
    from .pressure import PressureProcessor
    from .velocity_potential import VelocityPotentialProcessor
    
    pipeline = ProcessorPipeline()
    pipeline.add(PressureProcessor())
    pipeline.add(VelocityPotentialProcessor())
    
    return pipeline
