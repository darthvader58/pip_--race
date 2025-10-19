import React from 'react';
import CarCard from '../components/CarCard';
import RaceInfo from '../components/RaceInfo';

export default function LiveRacePage({ williamsData }) {
  return (
    <div className="page live-race-page">
      <h1 className="page-title">Monaco GP 2023 - Live Race</h1>
      
      <div className="cars-grid">
        <CarCard car={williamsData.car23} />
        <CarCard car={williamsData.car2} />
      </div>
      
      <RaceInfo />
    </div>
  );
}