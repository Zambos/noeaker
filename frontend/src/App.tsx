import React, { useState, useEffect } from 'react';
import TaskList from './components/TaskList';
import TaskForm from './components/TaskForm';
import MindMap from './components/MindMap';

// Define the structure of a task
interface Task {
  id: number;
  title: string;
}

const App: React.FC = () => {
  const [tasks, setTasks] = useState<Task[]>([]);

  useEffect(() => {
    fetch('http://localhost:8000/tasks')
      .then((response) => response.json())
      .then((data) => setTasks(data));
  }, []);

  const addTask = (task: Task) => {
    setTasks([...tasks, task]);
  };

  return (
    <div>
      <h1>ADHD Task Manager</h1>
      <TaskForm addTask={addTask} />
      <TaskList tasks={tasks} />
      <MindMap tasks={tasks} />
    </div>
  );
};

export default App;
