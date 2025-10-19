import React from 'react';

export default function AboutPage() {
  return (
    <div className="page about-page">
      <h1 className="page-title">About Williams Racing</h1>
      
      <div className="about-content">
        <section className="about-section">
          <h2 className="section-title">Our Mission</h2>
          <p className="section-text">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
            incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
            exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
            irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
            pariatur.
          </p>
        </section>
        
        <section className="about-section">
          <h2 className="section-title">Our Vision</h2>
          <p className="section-text">
            Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu 
            fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in 
            culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde 
            omnis iste natus error sit voluptatem accusantium doloremque laudantium.
          </p>
        </section>
        
        <section className="about-section">
          <h2 className="section-title">Team Excellence</h2>
          <p className="section-text">
            Totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto 
            beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit 
            aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione 
            voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor 
            sit amet, consectetur, adipisci velit.
          </p>
        </section>
      </div>
    </div>
  );
}