import React from 'react';

interface Task {
  id: number;
  title: string;
}

interface MindMapProps {
  tasks: Task[];
}

const MindMap: React.FC<MindMapProps> = ({ tasks }) => {
  return (
    <div>
      <h2>Mind Map (Placeholder)</h2>
      <p>{JSON.stringify(tasks, null, 2)}</p>
    </div>
  );
};

export default MindMap;
