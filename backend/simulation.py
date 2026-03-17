import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
import numpy as np
import random

class EnemyAgent(Agent):
    """Враждебная сущность (хищник), охотящаяся на роботов."""
    def __init__(self, model):
        super().__init__(model)
        self.speed = random.uniform(0.8, 1.5) # Even faster predators

    def step(self):
        target = self.sense_robot()
        if target and target.pos is not None:
            dx, dy = target.pos[0] - self.pos[0], target.pos[1] - self.pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.5:
                # Охота
                next_x = self.pos[0] + (dx/dist)*self.speed
                next_y = self.pos[1] + (dy/dist)*self.speed
                # clamping
                new_pos = (max(0, min(self.model.space.width-0.1, next_x)), 
                           max(0, min(self.model.space.height-0.1, next_y)))
                self.model.space.move_agent(self, new_pos)
            
            # Атака при контакте
            if dist < 1.5:
                target.energy -= 10  # More damage (from 2 to 10)
        else:
            # Случайное блуждание с ограничением границ
            next_x = self.pos[0] + random.uniform(-self.speed, self.speed)
            next_y = self.pos[1] + random.uniform(-self.speed, self.speed)
            new_pos = (max(0, min(self.model.space.width-0.1, next_x)), 
                       max(0, min(self.model.space.height-0.1, next_y)))
            self.model.space.move_agent(self, new_pos)

    def sense_robot(self):
        if self.pos is None: return None
        neighbors = self.model.space.get_neighbors(self.pos, 15.0, False)
        robots = [n for n in neighbors if isinstance(n, RobotAgent) and n.pos is not None]
        if not robots: return None
        return min(robots, key=lambda r: self.model.space.get_distance(self.pos, r.pos))

class Nest(Agent):
    """Общая база-гнездо колонии."""
    def __init__(self, model):
        super().__init__(model)
        self.resources_stored = 0
        self.total_collected = 0
        self.level = 1

    def step(self):
        # Гнездо растет при накоплении ресурсов (квадратичная сложность)
        upgrade_cost = self.level * 250
        if self.resources_stored >= upgrade_cost:
            self.level += 1
            # При росте уровня могут спавниться новые роботы бесплатно
            if self.model.robot_count < 100:
                new_robot = RobotAgent(self.model)
                self.model.schedule.add(new_robot)
                self.model.space.place_agent(new_robot, self.pos)
                self.model.robot_count += 1
                # Сбрасываем ресурсы: уровень * 300
                cost = (self.level * 300) 
                self.resources_stored = max(0, self.resources_stored - cost)

class Resource(Agent):
    """Агент-ресурс, который роботы должны найти."""
    def __init__(self, model):
        super().__init__(model)
        self.amount = random.randint(10, 50)

class RobotAgent(Agent):
    """Робот-агент с ролевой специализацией и ИИ."""
    def __init__(self, model, dna=None, role=None):
        super().__init__(model)
        self.energy = 100
        self.state = "exploring"
        self.carrying_resource = 0
        self.known_resource_pos = None
        self.is_shouting = False
        
        # Определяем роль
        if role:
            self.role = role
        else:
            self.role = random.choice(["scout", "worker", "hauler"])

        # DNA traits based on role
        if self.role == "scout":
            self.dna = dna or {
                "speed": random.uniform(2.5, 3.5),
                "vision": 40.0,
                "capacity": 10,
                "color_mod": 0 # Cyan
            }
        elif self.role == "hauler":
            self.dna = dna or {
                "speed": random.uniform(0.6, 1.2),
                "vision": 15.0,
                "capacity": 100,
                "color_mod": 1 # Yellow-ish
            }
        else: # worker
            self.dna = dna or {
                "speed": random.uniform(1.2, 2.0),
                "vision": 25.0,
                "capacity": 30,
                "color_mod": 2 # Standard
            }
        
        # Simple MLP weights (Input: 6 -> Hidden: 8 -> Output: 2)
        self.weights1 = np.random.uniform(-1, 1, (6, 8))
        self.weights2 = np.random.uniform(-1, 1, (8, 2))

    def step(self):
        if self.pos is None: return
        self.is_shouting = False
        # 1. Decision: go to nest or find resource?
        nest_pos = self.model.nest.pos
        if nest_pos is None: return
        target_pos = None
        
        # Haulers delivering more aggressively when half-full
        delivery_threshold = self.dna["capacity"] * 0.7
        if self.energy > 150 or self.carrying_resource >= delivery_threshold:
            self.state = "delivering"
            target_pos = nest_pos
        else:
            target_pos = self.sense_resource()
            # Если сами не видим, используем информацию от сородичей
            if not target_pos and self.known_resource_pos:
                target_pos = self.known_resource_pos
                # Если пришли в точку, а там пусто - забываем
                dist_to_info = np.sqrt((self.pos[0]-target_pos[0])**2 + (self.pos[1]-target_pos[1])**2)
                if dist_to_info < 3.0:
                    self.known_resource_pos = None
        
        # 2. Social Interaction: Collaboration
        # Разведчик "кричит" и притягивает других, если нашел ресурс
        if target_pos and self.role == "scout" and self.state != "delivering":
            self.share_knowledge(target_pos)

        # 3. Competition & Resource Check
        nearby_agents = self.model.space.get_neighbors(self.pos, 2.0, False)
        robots_nearby = [n for n in nearby_agents if isinstance(n, RobotAgent) and n != self]
        
        if len(robots_nearby) > 2:
            self.energy -= 0.5
            if self.state != "delivering": self.state = "competing"

        # Check for resource eating (reuse nearby_agents)
        for neighbor in nearby_agents:
            if isinstance(neighbor, Resource):
                self.energy += neighbor.amount
                self.model.kill_agent(neighbor)
                self.state = "harvesting"
                break

        dx_res, dy_res = (0, 0)
        if target_pos and self.state != "delivering":
            dx_res, dy_res = target_pos[0] - self.pos[0], target_pos[1] - self.pos[1]
            dist = np.sqrt(dx_res**2 + dy_res**2)
            dx_res, dy_res = dx_res/max(0.1, dist), dy_res/max(0.1, dist)
        
        dx_nest, dy_nest = nest_pos[0] - self.pos[0], nest_pos[1] - self.pos[1]
        dist_nest = np.sqrt(dx_nest**2 + dy_nest**2)
        dx_nest, dy_nest = dx_nest/max(1, dist_nest), dy_nest/max(1, dist_nest)

        inputs = np.array([dx_res, dy_res, dx_nest, dy_nest, self.energy/100, self.dna["speed"]])
        
        h = np.tanh(np.dot(inputs, self.weights1))
        outputs = np.tanh(np.dot(h, self.weights2))
        
        # Базовое движение от нейросети
        # Если есть target_pos (из зрения или памяти), идем к нему
        if target_pos:
            dx, dy = target_pos[0] - self.pos[0], target_pos[1] - self.pos[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                move_x = (dx/dist) * self.dna["speed"]
                move_y = (dy/dist) * self.dna["speed"]
        else:
            move_x = outputs[0] * self.dna["speed"]
            move_y = outputs[1] * self.dna["speed"]

        # Если цели нет, добавляем блуждание (Random Walk)
        if dx_res == 0 and dy_res == 0 and self.state != "delivering":
            move_x += random.uniform(-0.5, 0.5)
            move_y += random.uniform(-0.5, 0.5)
        
        # Добавляем микро-джиттер, чтобы не "зависать" на месте при малых весах
        move_x += random.uniform(-0.05, 0.05)
        move_y += random.uniform(-0.05, 0.05)
        
        # Ensure new_pos is within bounds and not None
        new_x = max(0, min(self.model.space.width - 0.1, self.pos[0] + move_x))
        new_y = max(0, min(self.model.space.height - 0.1, self.pos[1] + move_y))
        
        self.model.space.move_agent(self, (new_x, new_y))
        
        # Reproduction and status checks MUST happen while agent is alive
        if self.pos is not None:
            self.check_reproduction()
        if self.pos is not None:
            self.check_nest_delivery()
        
        self.consume_energy() # This can kill 'self'

    def check_nest_delivery(self):
        """Сдача ресурсов в гнездо."""
        dist = self.model.space.get_distance(self.pos, self.model.nest.pos)
        if dist < 3.0:
            if self.carrying_resource > 0:
                self.model.nest.resources_stored += self.carrying_resource
                self.model.nest.total_collected += self.carrying_resource
                self.carrying_resource = 0
            # Робот обменивает лишнюю энергию на прогресс гнезда
            # Робот обменивает лишнюю энергию (выше 120) и все ресурсы на прогресс гнезда
            if self.energy > 120:
                transfer = self.energy - 100
                self.model.nest.resources_stored += (transfer + self.carrying_resource)
                self.carrying_resource = 0
                self.energy = 100
            elif self.carrying_resource > 0:
                self.model.nest.resources_stored += self.carrying_resource
                self.carrying_resource = 0
            
            self.state = "exploring"

    def check_reproduction(self):
        """Размножение при избытке энергии (деление)."""
        if self.energy >= 200 and self.model.robot_count < 100:
            self.energy /= 2
            # Наследуем роль или мутируем
            offspring_role = self.role
            if random.random() < 0.1: # 10% шанс мутации роли
                offspring_role = random.choice(["scout", "worker", "hauler"])
                
            offspring = RobotAgent(self.model, dna=self.dna.copy(), role=offspring_role)
            offspring.energy = self.energy
            self.model.schedule.add(offspring)
            self.model.space.place_agent(offspring, self.pos)
            self.state = "exploring"
            self.model.robot_count += 1
            self.model.total_reproductions += 1

    def share_knowledge(self, resource_pos):
        """Передача информации о ресурсе ближайшим соседям (Сотрудничество)."""
        self.is_shouting = True
        # Разведчики "кричат" громче (радиус 30), остальные тише (15)
        broadcast_range = 30.0 if self.role == "scout" else 15.0
        neighbors = self.model.space.get_neighbors(self.pos, broadcast_range, False)
        for n in neighbors:
            if isinstance(n, RobotAgent) and n != self:
                # Особенно важно передавать инфу тяжелым грузчикам
                n.known_resource_pos = resource_pos

    def sense_resource(self):
        """Зрение агента: поиск ближайшего ресурса в радиусе зрения (из DNA)."""
        if self.pos is None: return None
        neighbors = self.model.space.get_neighbors(self.pos, self.dna["vision"], False)
        resources = [n for n in neighbors if isinstance(n, Resource) and n.pos is not None]
        if not resources:
            return None
        # Возвращаем позицию ближайшего
        closest = min(resources, key=lambda r: self.model.space.get_distance(self.pos, r.pos))
        return closest.pos

    def move(self):
        pass # Logic moved to step()

    def consume_energy(self):
        self.energy -= 0.1 * self.dna["speed"]
        if self.energy <= 0:
            self.model.kill_agent(self)

    def check_resources(self):
        nearby_agents = self.model.space.get_neighbors(self.pos, 2.0, False)
        for neighbor in nearby_agents:
            if isinstance(neighbor, Resource):
                self.carrying_resource += neighbor.amount
                self.model.kill_agent(neighbor)
                self.state = "harvesting"
                break

class SwarmEnvironment(Model):
    def __init__(self, width=100, height=100, initial_robots=50, initial_resources=30, initial_enemies=8):
        super().__init__()
        self.space = ContinuousSpace(width, height, False)
        self.schedule = RandomActivation(self)
        self.running = True
        self.colony_age = 0
        self.total_reproductions = 0
        self.robot_count = initial_robots

        # Создаем Гнездо в центре
        self.nest = Nest(self)
        self.schedule.add(self.nest)
        self.space.place_agent(self.nest, (width/2, height/2))

        # Создаем роботов
        for i in range(initial_robots):
            robot = RobotAgent(self)
            self.schedule.add(robot)
            x = self.random.random() * width
            y = self.random.random() * height
            self.space.place_agent(robot, (x, y))

        # Создаем ресурсы
        for i in range(initial_resources):
            res = Resource(self)
            self.schedule.add(res)
            valid_pos = False
            while not valid_pos:
                x = self.random.random() * width
                y = self.random.random() * height
                dist_to_nest = np.sqrt((x - width/2)**2 + (y - height/2)**2)
                if dist_to_nest > 35:
                    valid_pos = True
            self.space.place_agent(res, (x, y))

        # Создаем врагов
        for i in range(initial_enemies):
            enemy = EnemyAgent(self)
            self.schedule.add(enemy)
            x = self.random.random() * width
            y = self.random.random() * height
            self.space.place_agent(enemy, (x, y))

    def step(self):
        # Кэшируем количество роботов и ролей для оптимизации
        all_robots = [a for a in self.schedule.agents if isinstance(a, RobotAgent)]
        self.robot_count = len(all_robots)
        self.role_counts = {
            "scout": len([r for r in all_robots if r.role == "scout"]),
            "hauler": len([r for r in all_robots if r.role == "hauler"]),
            "worker": len([r for r in all_robots if r.role == "worker"])
        }
        self.colony_age += 1
        
        # Гарантированный спавн если колония критически уменьшилась
        if self.robot_count < 15: 
            print(f"COLONY WEAK ({self.robot_count}): Emergency spawning reinforcements...", flush=True)
            # Спавним полный отряд специалистов
            new_squad = (
                ["scout"] * 3 +   # 3 разведчика
                ["hauler"] * 2 +  # 2 грузчика
                ["worker"] * 5    # 5 рабочих
            )
            for role in new_squad:
                r = RobotAgent(self, role=role)
                self.schedule.add(r)
                self.space.place_agent(r, self.nest.pos)
            self.robot_count += 10

        self.schedule.step()

    def kill_agent(self, agent):
        """Безопасное удаление агента."""
        try:
            if agent in self.schedule.agents:
                self.schedule.remove(agent)
            if hasattr(agent, "pos") and agent.pos is not None:
                self.space.remove_agent(agent)
        except Exception as e:
            pass # Игнорируем ошибки при повторном удалении
            
        # Если это был ресурс, спавним новый
        if isinstance(agent, Resource):
            new_res = Resource(self)
            self.schedule.add(new_res)
            # Спавним ресурсы подальше от гнезда (минимум 35 единиц дистанции)
            valid_pos = False
            while not valid_pos:
                x = self.random.random() * self.space.width
                y = self.random.random() * self.space.height
                dist_to_nest = np.sqrt((x - self.nest.pos[0])**2 + (y - self.nest.pos[1])**2)
                if dist_to_nest > 35:
                    valid_pos = True
            self.space.place_agent(new_res, (x, y))
        elif isinstance(agent, RobotAgent):
            self.robot_count -= 1
