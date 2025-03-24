import os
import datetime
import json
import warnings

try:
    from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, create_engine
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import relationship, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    warnings.warn(
        "SQLAlchemy not found. Bible storage will use a simplified in-memory model. "
        "Install SQLAlchemy with 'pip install sqlalchemy' for full functionality."
    )
    SQLALCHEMY_AVAILABLE = False
    # Create placeholder classes
    class Column:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class Integer:
        pass

    class String:
        pass

    class JSON:
        pass

    class ForeignKey:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class DateTime:
        pass

    def create_engine(*args, **kwargs):
        return None

    def declarative_base():
        return type('Base', (), {'__tablename__': ''})

    class relationship:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class sessionmaker:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
        
        def __call__(self):
            return type('Session', (), {
                'add': lambda x: None,
                'commit': lambda: None,
                'query': lambda x: [],
                'delete': lambda x: None,
                '__enter__': lambda x: x,
                '__exit__': lambda x, *args: None
            })()

# Create base model
Base = declarative_base()

class Bible(Base):
    """Base class for all Bible entries"""
    __tablename__ = 'bibles'
    
    id = Column(Integer, primary_key=True)
    type = Column(String, nullable=False)  # 'character', 'plot', 'setting', 'theme'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Type-specific data stored as JSON
    content = Column(JSON, nullable=False)
    
    # Store version history as JSON array of diffs
    version_history = Column(JSON, default=lambda: json.dumps([]))
    
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'bible'
    }


class CharacterBible(Bible):
    """Character bible entries containing character details"""
    __mapper_args__ = {
        'polymorphic_identity': 'character'
    }


class PlotBible(Bible):
    """Plot bible entries containing scene and act information"""
    __mapper_args__ = {
        'polymorphic_identity': 'plot'
    }


class SettingBible(Bible):
    """Setting bible entries containing location details"""
    __mapper_args__ = {
        'polymorphic_identity': 'setting'
    }


class ThemeBible(Bible):
    """Theme bible entries containing thematic elements"""
    __mapper_args__ = {
        'polymorphic_identity': 'theme'
    }


def get_engine(db_path='script_gen.db'):
    """Create a database engine"""
    if not SQLALCHEMY_AVAILABLE:
        warnings.warn("SQLAlchemy not available; using mock engine")
        return None
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    engine = create_engine(f'sqlite:///{db_path}')
    return engine


def initialize_database(engine):
    """Create all tables in the database"""
    if not SQLALCHEMY_AVAILABLE:
        warnings.warn("SQLAlchemy not available; using mock session")
        return sessionmaker()
        
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine) 