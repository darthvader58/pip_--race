import React from 'react';

export default function Navigation({ currentRoute, onNavigate, connected, error }) {
  const links = [
    { path: '/', label: 'Live Race' },
    { path: '/strategy', label: 'Strategy' },
    { path: '/about', label: 'About' }
  ];

  return (
    <nav className="navigation">
      <div className="nav-brand">
        <span className="williams-logo">WILLIAMS RACING</span>
        <span className="connection-status">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`} />
          {error ? (
            <span className="status-error" title={error}>
              ERROR
            </span>
          ) : connected ? (
            'LIVE'
          ) : (
            'CONNECTING...'
          )}
        </span>
      </div>
      
      <div className="nav-links">
        {links.map(link => (
          <button
            key={link.path}
            onClick={() => onNavigate(link.path)}
            className={`nav-link ${currentRoute === link.path ? 'active' : ''}`}
          >
            {link.label}
          </button>
        ))}
      </div>
    </nav>
  );
}