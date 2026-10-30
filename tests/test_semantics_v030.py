"""Offline semantic regressions for the five-tier audit; no live model or web."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode

from schain.kg.entity_types import company_type_hint
from schain.kg.relation_evidence import (
    check_supplies_to_direction, classify_supply_direction, lexical_relation_gate,
    load_relation_evidence, UPSTREAM_EXPLICIT, DOWNSTREAM_EXPLICIT, DIRECTION_AMBIGUOUS,
)
from schain.kg.resolve import identity_plausible, resolve_entities, sanitize_llm_groups
from schain.kg.validate import validate_facts_frame
from schain.names import canonical_id, mention_key


@pytest.mark.parametrize('evidence', [
    'CATL supplies BMW.', 'CATL is a supplier to BMW.',
    'BMW selects CATL as supplier.', 'BMW selected CATL as a battery supplier.',
    'BMW sources battery cells from CATL.', 'BMW purchases battery cells from CATL.',
    'BMW buys lithium-ion battery cells from CATL.', 'CATL delivers battery cells to BMW.',
    'CATL supplies battery cells and modules to BMW.', 'CATL will supply BMW.',
    'BMW mentioned CATL. BMW later announced a plan. CATL supplies BMW.',
])
def test_explicit_roles_allow_inserted_products(evidence):
    assert check_supplies_to_direction('CATL', 'BMW', evidence)[0]
    assert not check_supplies_to_direction('BMW', 'CATL', evidence)[0]


@pytest.mark.parametrize('evidence', [
    'CATL partners with BMW.', 'CATL collaborates with BMW.',
    'CATL entered a supply agreement with BMW.', 'CATL signed a strategic pact with BMW.',
    'BMW selected CATL as a research partner.', 'CATL never supplies BMW.',
    'CATL does not supply BMW.', 'CATL supplies Tesla. BMW supplies Tesla.',
    'CATL supplies BMWbrakes.', 'CATL supplies BMW i7.',
    'CATL supplies cells to Tesla and Bosch supplies cells to BMW.', 'If CATL supplies BMW, production could begin.',
])
def test_ambiguity_negation_and_cross_sentence_roles_do_not_pass(evidence):
    assert check_supplies_to_direction('CATL', 'BMW', evidence) == (False, 'ambiguous_supply_direction')


def test_classifier_uses_local_upstream_priority_in_mixed_article():
    assert classify_supply_direction('BorgWarner', 'BorgWarner supplies Stellantis.') == DOWNSTREAM_EXPLICIT
    assert classify_supply_direction('BorgWarner', 'Aster supplies BorgWarner.') == UPSTREAM_EXPLICIT
    assert classify_supply_direction('BorgWarner', 'BorgWarner signs a supply agreement with Aster.') == DIRECTION_AMBIGUOUS
    assert classify_supply_direction('BorgWarner', 'BorgWarner supplies Stellantis. Aster supplies BorgWarner.') == UPSTREAM_EXPLICIT


def test_active_yaml_direction_templates_and_wildcard_cues_are_executable(tmp_path):
    policy = {'suppliesTo': {'positive': ['transfers .* into'],
        'direction_templates_supplier_first': ['{subject} transfers .* into {object}']}}
    path = tmp_path / 'policy.yaml'
    path.write_text(yaml.safe_dump(policy))
    loaded = load_relation_evidence(str(path))
    evidence = 'Aster transfers electric motor parts into BMW.'
    assert lexical_relation_gate('suppliesTo', evidence, policy=loaded)[0]
    assert check_supplies_to_direction('Aster', 'BMW', evidence, policy=loaded)[0]
    assert not check_supplies_to_direction('Aster', 'BMW', evidence)[0]
    assert classify_supply_direction('BMW', evidence, policy=loaded) == UPSTREAM_EXPLICIT
    path.write_text(yaml.safe_dump({'suppliesTo': {}}))
    assert not check_supplies_to_direction('Aster', 'BMW', evidence, policy=load_relation_evidence(str(path)))[0]


@pytest.mark.parametrize('name', ['BMW i7', 'BMW iX3', 'Tesla Model 3', 'Rivian R1S'])
def test_model_structure_is_product_before_resolution_cache(name):
    graph = Neo4jGraph(nodes=[Neo4jNode(id='n', label='Company', properties={'name': name})])
    resolution, entries = resolve_entities(graph)
    assert resolution['n']['label'] == 'Product'
    assert resolution['n']['canonical_id'].startswith('product:')
    assert entries[0]['method'].startswith('type_repair:')
    assert company_type_hint(name)[0] == 'Product'


def _fact(subject='CATL', obj='BMW', evidence='CATL supplies BMW.', relation='suppliesTo'):
    return {'doc_id': 'doc', 'fact_id': 'f', 'chunk_index': 0,
        'subject_id': canonical_id('Company', subject), 'subject_name': subject,
        'subject_type': 'Company', 'subject_mention': subject, 'relation': relation,
        'object_id': canonical_id('Company', obj), 'object_name': obj,
        'object_type': 'Company', 'object_mention': obj, 'evidence': evidence}


def test_mislabeled_vehicle_and_facility_cannot_enter_company_graph(ingest_config, ontology):
    for obj, reason in [('BMW i7', 'branded_model_structure'), ('BMW Plant Landshut', 'facility_not_company')]:
        evidence = f'CATL supplies {obj}.'
        accepted, rejected = validate_facts_frame(pd.DataFrame([_fact(obj=obj,evidence=evidence)]),
            config=ingest_config, ontology=ontology, document_text=evidence)
        assert accepted.empty
        assert rejected.iloc[0].rejection_stage == 'entity_type'
        assert reason in rejected.iloc[0].rejection_reason


@pytest.mark.parametrize('left,right', [
    ('Thyssenkrupp','Thyssenkrupp Steel'), ('Thyssenkrupp','thyssenkrupp nucera'),
    ('Thyssenkrupp','ThyssenKrupp Aerospace'), ('Thyssenkrupp','thyssenkrupp rothe erde'),
    ('CATL Ningde','Contemporary Amperex Technology Co. Limited'),
    ('BYD','CATL'), ('Blade Batteries','Ultium'),
    ('ventilated rotors','carbon-sleeved rotors'),
])
def test_distinct_identity_pairs_and_unsafe_cache_stay_separate(left,right):
    assert not identity_plausible(left,right)
    graph=Neo4jGraph(nodes=[Neo4jNode(id='n',label='Company',properties={'name':left})])
    cached={mention_key('Company',left): {'label':'Company','canonical_name':right,
        'canonical_id':canonical_id('Company',right),'method':'llm'}}
    resolution,_=resolve_entities(graph,cache=cached)
    assert resolution['n']['canonical_id'] != canonical_id('Company',right)


def test_sanitized_groups_have_no_unsafe_transitive_pairs():
    groups=sanitize_llm_groups([['ABC','Alpha Beta Chemicals','Alpha Beta Chemicals Subsidiary']])
    assert len(groups)>1
    for group in groups:
        assert all(identity_plausible(a,b) for a in group for b in group)


class EntailmentLLM:
    def __init__(self, fail=False):
        self.prompts=[]
        self.fail=fail
    def invoke(self,prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise ConnectionError('offline stub outage')
        candidates=[json.loads(line) for line in prompt.splitlines() if line.startswith('{"candidate_fact_id"')]
        return SimpleNamespace(content=json.dumps({'results':[
            {'candidate_fact_id':row['candidate_fact_id'],'decision':'supports'} for row in candidates]}))


def test_custom_policy_reaches_entailment_prompt(ingest_config, ontology, tmp_path):
    config=copy.deepcopy(ingest_config)
    policy=copy.deepcopy(load_relation_evidence())
    policy['suppliesTo']['description']='CUSTOM_ROLE_POLICY_SENTINEL'
    path=tmp_path/'policy.yaml'; path.write_text(yaml.safe_dump(policy))
    config.setdefault('validation',{}).setdefault('evidence',{})['policy_file']=str(path)
    llm=EntailmentLLM()
    accepted,rejected=validate_facts_frame(pd.DataFrame([_fact()]),config=config,ontology=ontology,
        llm=llm,document_text='CATL supplies BMW.')
    assert len(accepted)==1 and rejected.empty
    assert 'CUSTOM_ROLE_POLICY_SENTINEL' in llm.prompts[0]
    assert 'supply agreement with' in llm.prompts[0]


def test_model_transport_failure_does_not_become_semantic_rejection(ingest_config, ontology):
    with pytest.raises(RuntimeError,match='retryable'):
        validate_facts_frame(pd.DataFrame([_fact()]),config=ingest_config,ontology=ontology,
            llm=EntailmentLLM(fail=True),document_text='CATL supplies BMW.')


def test_resolver_repaired_vehicle_cannot_trigger_fabricated_relation_repair(ingest_config, ontology):
    row = _fact(obj="BMW i7", evidence="CATL supplies BMW i7.")
    row["object_type"] = "Product"
    row["object_id"] = canonical_id("Product", "BMW i7")
    class NoCall:
        def invoke(self, prompt):
            raise AssertionError("a vehicle supply role is already deterministically invalid")
    accepted, rejected = validate_facts_frame(pd.DataFrame([row]), config=ingest_config,
        ontology=ontology, llm=NoCall(), document_text=row["evidence"])
    assert accepted.empty
    assert rejected.iloc[0].rejection_reason == "invalid_product_supply_role:object"
