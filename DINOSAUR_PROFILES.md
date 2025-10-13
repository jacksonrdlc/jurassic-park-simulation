# Dinosaur Behavioral Profiles

Based on paleontological research, this document defines realistic behavioral characteristics for each dinosaur species in the simulation.

---

## HERBIVORES

### Stegosaurus (Plated Herbivore)
**Time Period:** Late Jurassic (155-145 MYA)
**Size:** 21-30 feet long, 12 feet tall, ~2 tons
**Sprite:** `Stegosaurus_32x32.png`

**Habitat Preferences:**
- Subtropical forests with undergrowth
- Prefers dense vegetation areas
- Avoids steep terrain (cannot walk fast)

**Diet:**
- Low-browsing herbivore
- Eats mosses, ferns, horsetails, cycads, conifers
- Small peg-shaped teeth for grinding soft plants
- Requires frequent feeding due to large size

**Social Structure:**
- Lives in family herds (4-10 individuals)
- Multi-generational groups with juveniles and adults
- Moderate herd cohesion - stays within 5-10 cells of herd

**Behavior:**
- **Speed:** Very slow (max 9.5-11 mph) - 1 cell/step
- **Defense:** 70% defense rating (tail spikes are highly effective)
- **Temperament:** Defensive when threatened, otherwise peaceful
- **Special:** When threatened, has 30% chance to counterattack with tail spikes

**Metabolism:**
- 1.6x energy consumption (large size)
- Requires 120 energy to reproduce

---

### Pachycephalosaurus (Dome-Headed Herbivore)
**Time Period:** Late Cretaceous (70-66 MYA)
**Size:** 15 feet long, ~820-990 lbs
**Sprite:** `Pachycephalosaurus_16x16.png`

**Habitat Preferences:**
- Coastal plains to dense forests
- Subtropical warm humid environments
- Prefers open areas with flowering plants

**Diet:**
- Mixed diet: leaves, seeds, fruits
- Sharp serrated teeth for shredding plants
- May occasionally eat insects/small prey (omnivorous tendency)
- Moderate feeding frequency

**Social Structure:**
- Small herds (3-6 individuals) OR solitary
- Males establish dominance hierarchies
- Territorial during breeding season

**Behavior:**
- **Speed:** Medium-fast (bipedal) - 2 cells/step
- **Defense:** 45% defense rating (head-butting, intimidation displays)
- **Temperament:** Aggressive when defending territory
- **Special:** When two Pachycephalosaurus meet, 20% chance of territorial display/head-butt contest

**Metabolism:**
- 0.9x energy consumption (medium-sized, bipedal)
- Requires 90 energy to reproduce

---

### Brachiosaurus (Massive Sauropod)
**Time Period:** Late Jurassic (155.6-145.5 MYA)
**Size:** Up to 85 feet long, ~56 tons, 30ft browsing height
**Sprite:** `Brachiosaurus_32x32.png`

**Habitat Preferences:**
- Flat plains and lowlands (avoids hills due to energy cost)
- Areas with tall vegetation
- Prefers open savannas with scattered trees

**Diet:**
- High-browser (eats from treetops up to 30ft high)
- Massive consumption requirements
- Can reach vegetation other herbivores cannot
- Requires constant feeding

**Social Structure:**
- Mixed-age herds (8-15 individuals)
- Juveniles stay with adults for protection
- Strong herd cohesion - stays within 8-12 cells of group
- May communicate with low-frequency calls

**Behavior:**
- **Speed:** Very slow (massive size) - 1 cell/step, cannot run
- **Defense:** 85% defense rating (massive size intimidates predators)
- **Temperament:** Peaceful, relies on size for protection
- **Special:** Size discourages most predators - carnivores <50 energy won't attack

**Metabolism:**
- 2.2x energy consumption (MASSIVE size)
- Requires 180 energy to reproduce
- Energy drains faster when moving uphill

---

### Archeopteryx (Proto-Bird)
**Time Period:** Late Jurassic (150 MYA)
**Size:** Magpie to raven-sized, ~18 inches long
**Sprite:** `Archeopteryx_16x16.png`

**Habitat Preferences:**
- Forested areas near water
- Subtropical islands with lagoons
- Perches in trees (prefers forest terrain)
- At home both on ground and in trees

**Diet:**
- Insectivore (primarily eats insects)
- May eat small lizards and invertebrates
- Requires frequent small meals
- Fast metabolism

**Social Structure:**
- Mostly solitary or pairs
- May form loose flocks of 3-5 near abundant food
- Low herd cohesion - independent behavior

**Behavior:**
- **Speed:** Fast (3 cells/step - can glide/fly short distances)
- **Defense:** 15% defense rating (small, relies on escape)
- **Temperament:** Skittish, flees from danger
- **Special:** When threatened, has 40% chance to "glide" 5 cells away instantly

**Metabolism:**
- 0.5x energy consumption (very small, efficient)
- Requires only 40 energy to reproduce
- Diurnal - more active during "day" cycles

---

## EXISTING SPECIES (Updated)

### Triceratops (Already Implemented)
- 50% defense, 1 cell/step speed
- Small herds, heavily armored
- 1.5x metabolism

### Gallimimus (Already Implemented)
- 30% defense, 3 cells/step speed (fastest)
- Small flocks, relies on speed
- 0.7x metabolism (very efficient)

### T-Rex (Already Implemented)
- Apex predator, 100% attack success
- 1 cell/step speed (ambush hunter)
- 1.8x metabolism (very hungry)

### Velociraptor (Already Implemented)
- Pack hunter, 50% attack success
- 2 cells/step speed
- Pack hunting bonus: 1.5x
- 0.8x metabolism

---

## GAMEPLAY IMPLICATIONS

### Herd Mechanics
Herbivores will seek to stay near others of their species:
- **Strong Herd:** Brachiosaurus, Stegosaurus (stay within 8-10 cells)
- **Moderate Herd:** Triceratops, Pachycephalosaurus (stay within 5-7 cells)
- **Weak Herd:** Gallimimus, Archeopteryx (independent behavior)

### Predator-Prey Dynamics
- Archeopteryx: Too small/fast for most predators, ignored by T-Rex
- Pachycephalosaurus: Will fight back occasionally (territorial)
- Brachiosaurus: Size deters attacks from weak carnivores
- Stegosaurus: Dangerous to attack (tail spike counterattacks)

### Terrain Preferences
- **Forest:** Archeopteryx (trees), Stegosaurus (undergrowth)
- **Grassland:** All herbivores graze here
- **Plains:** Brachiosaurus prefers flat open areas
- **Rainforest:** Pachycephalosaurus likes dense humid zones

---

## IMPLEMENTATION NOTES

Each species should have:
1. **Speed** modifier (cells per step)
2. **Defense** rating (% chance to survive attack)
3. **Metabolism** multiplier (energy consumption rate)
4. **Herd behavior** (cohesion radius and strength)
5. **Habitat preference** (terrain type attractiveness)
6. **Special abilities** (counter-attack, gliding, etc.)
7. **Size class** (affects predator targeting)

These profiles provide the foundation for implementing realistic dinosaur behaviors in the simulation.
