#include "scenario_registry.h"

namespace rocketsim_cuda {

ScenarioRegistry& ScenarioRegistry::Instance() {
    static ScenarioRegistry instance;
    return instance;
}

void ScenarioRegistry::Register(std::shared_ptr<IScenario> scenario) {
    if (!scenario) return;
    const std::string& name = scenario->GetName();
    if (scenarios_.find(name) == scenarios_.end()) {
        scenario_order_.push_back(name);
    }
    scenarios_[name] = scenario;
}

void ScenarioRegistry::RegisterAlias(const std::string& alias, const std::string& target_name) {
    auto it = scenarios_.find(target_name);
    if (it != scenarios_.end()) {
        scenarios_[alias] = it->second;
    }
}

std::shared_ptr<IScenario> ScenarioRegistry::Get(const std::string& name) const {
    auto it = scenarios_.find(name);
    if (it != scenarios_.end()) {
        return it->second;
    }
    return nullptr;
}

bool ScenarioRegistry::Has(const std::string& name) const {
    return scenarios_.find(name) != scenarios_.end();
}

std::vector<std::string> ScenarioRegistry::GetAllNames() const {
    return scenario_order_;
}

std::vector<std::shared_ptr<IScenario>> ScenarioRegistry::GetAll() const {
    std::vector<std::shared_ptr<IScenario>> list;
    list.reserve(scenario_order_.size());
    for (const auto& name : scenario_order_) {
        list.push_back(scenarios_.at(name));
    }
    return list;
}

void RegisterAllScenarios() {
    // Preserve canonical registration order
    RegisterCarScenarios();
    RegisterBounceScenarios();
    RegisterAblationScenarios();
    RegisterArenaScenarios();
    RegisterMultiCarScenarios();
    RegisterRandomScenarios();
}

} // namespace rocketsim_cuda
