import { useState } from "react";
import axios from "axios";
import { Plane, Search, Menu, User, MapPin } from "lucide-react";
import { motion } from "framer-motion";

const API_URL = "http://localhost:8000";

const DEFAULT_DATA = {
  host_since: "2020-01-01",
  host_response_time: "within an hour",
  host_response_rate: "100%",
  host_acceptance_rate: "95%",
  host_is_superhost: "t",
  host_listings_count: 1,
  host_total_listings_count: 1,
  host_has_profile_pic: "t",
  host_identity_verified: "t",
  neighbourhood_cleansed: "Palermo",
  latitude: -34.58,
  longitude: -58.42,
  property_type: "Entire rental unit",
  room_type: "Entire home/apt",
  accommodates: 2,
  bathrooms: 1.0,
  bedrooms: 1.0,
  beds: 1.0,
  price: "$50.00",
  minimum_nights: 2,
  maximum_nights: 30,
  minimum_minimum_nights: 2,
  maximum_minimum_nights: 2,
  minimum_maximum_nights: 1125,
  maximum_maximum_nights: 1125,
  minimum_nights_avg_ntm: 2.0,
  maximum_nights_avg_ntm: 1125.0,
  has_availability: "t",
  availability_30: 10,
  availability_60: 20,
  availability_90: 30,
  availability_365: 100,
  number_of_reviews: 10,
  number_of_reviews_ltm: 2,
  number_of_reviews_l30d: 0,
  review_scores_rating: 4.8,
  review_scores_accuracy: 4.9,
  review_scores_cleanliness: 4.8,
  review_scores_checkin: 4.9,
  review_scores_communication: 4.9,
  review_scores_location: 4.9,
  review_scores_value: 4.7,
  reviews_per_month: 0.5,
  amenities: '["Wifi", "Kitchen", "Air conditioning"]',
};

const LABELS_MAP = {
  0: "High",
  1: "Low",
  2: "Mid",
  3: "Zero",
};

const AMENITIES_LIST = [
  "Kitchen", "Wifi", "Hot Water", "Dishes And Silverware", "Cooking Basics", "Hangers", 
  "Essentials", "Bed Linens", "Refrigerator", "Hair Dryer", "Air Conditioning", 
  "Microwave", "Elevator", "Bidet", "Tv", "Dedicated Workspace", 
  "Room-Darkening Shades", "Hot Water Kettle", "Iron", "Extra Pillows And Blankets"
];



const COLORS_MAP = {
  0: "text-green-600",
  1: "text-orange-500",
  2: "text-blue-500",
  3: "text-gray-400",
};

function App() {
  const [formData, setFormData] = useState(DEFAULT_DATA);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleAmenityChange = (amenity) => {
    let current = [];
    try {
      current = JSON.parse(formData.amenities || "[]");
    } catch { 
      current = []; 
    }
    
    if (current.includes(amenity)) {
      current = current.filter(a => a !== amenity);
    } else {
      current.push(amenity);
    }
    setFormData(prev => ({ ...prev, amenities: JSON.stringify(current) }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        data: [formData],
      });
      setPrediction(response.data.predictions[0]);
    } catch (err) {
      console.error(err);
      setError(
        "Prediction failed. " + (err.response?.data?.detail || err.message)
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="border-b sticky top-0 bg-white z-50 px-6 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2 text-[--airbnb-red] font-bold text-xl">
          <Plane className="w-8 h-8" />
          <span>Airbnb Predictor</span>
        </div>

        <div className="hidden md:flex items-center shadow-sm border rounded-full py-2.5 px-6 gap-4 text-sm font-medium hover:shadow-md transition cursor-pointer">
          <span>Anywhere</span>
          <span className="border-l px-4">Any week</span>
          <span className="border-l px-4 text-gray-500 font-normal">
            Add guests
          </span>
          <div className="bg-[--airbnb-red] p-2 rounded-full text-white">
            <Search className="w-4 h-4" />
          </div>
        </div>

        <div className="flex items-center gap-4 border p-2 rounded-full hover:shadow-md transition cursor-pointer">
          <Menu className="w-4 h-4 ml-1" />
          <User className="w-8 h-8 fill-gray-500 text-gray-500 bg-gray-200 rounded-full p-1" />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 md:grid-cols-2 gap-12">
        {/* Left Col: Form */}
        <div className="space-y-8">
          <section>
            <h1 className="text-3xl font-bold mb-6 text-[#222]">
              Listing Details
            </h1>
            <div className="bg-white rounded-xl border p-6 space-y-6">


              <div className="space-y-4">
                <h3 className="font-semibold text-lg">Location & Type</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="input-group">
                    <label>Neighbourhood</label>
                    <input
                      className="input-control"
                      name="neighbourhood_cleansed"
                      value={formData.neighbourhood_cleansed}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="input-group">
                    <label>Property Type</label>
                    <select
                      className="input-control"
                      name="property_type"
                      value={formData.property_type}
                      onChange={handleChange}
                    >
                      <option>Entire rental unit</option>
                      <option>Private room in rental unit</option>
                    </select>
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="font-semibold text-lg">Details</h3>
                <div className="grid grid-cols-3 gap-4">
                  <div className="input-group">
                    <label>Accommodates</label>
                    <input
                      type="number"
                      className="input-control"
                      name="accommodates"
                      value={formData.accommodates}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="input-group">
                    <label>Bathrooms</label>
                    <input
                      type="number"
                      className="input-control"
                      name="bathrooms"
                      value={formData.bathrooms}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="input-group">
                    <label>Bedrooms</label>
                    <input
                      type="number"
                      className="input-control"
                      name="bedrooms"
                      value={formData.bedrooms}
                      onChange={handleChange}
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-4">
                <h3 className="font-semibold text-lg">
                  Pricing & Availability
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div className="input-group">
                    <label>Price</label>
                    <input
                      className="input-control"
                      name="price"
                      value={formData.price}
                      onChange={handleChange}
                    />
                  </div>
                  <div className="input-group">
                    <label>Min Nights</label>
                    <input
                      type="number"
                      className="input-control"
                      name="minimum_nights"
                      value={formData.minimum_nights}
                      onChange={handleChange}
                    />
                  </div>
                </div>
              </div>

               <div className="space-y-4">
                <h3 className="font-semibold text-lg">Occupancy Factors (Cold Start)</h3>
                <div className="grid grid-cols-2 gap-4">
                   <div className="input-group">
                      <label>Superhost</label>
                      <select className="input-control" name="host_is_superhost" value={formData.host_is_superhost} onChange={handleChange}>
                        <option value="t">Yes</option>
                        <option value="f">No</option>
                      </select>
                   </div>
                   <div className="input-group">
                       <label>Room Type</label>
                       <select className="input-control" name="room_type" value={formData.room_type} onChange={handleChange}>
                         <option>Entire home/apt</option>
                         <option>Private room</option>
                         <option>Shared room</option>
                         <option>Hotel room</option>
                       </select>
                   </div>
                   
                   <div className="input-group text-sm font-medium">
                      <label>Beds</label>
                      <input type="number" className="input-control" name="beds" value={formData.beds} onChange={handleChange} />
                   </div>
                   <div className="input-group text-sm font-medium">
                      <label>Bedrooms</label>
                      <input type="number" className="input-control" name="bedrooms" value={formData.bedrooms} onChange={handleChange} />
                   </div>

                   <div className="col-span-2">
                      <label className="block mb-2 font-medium">Amenities</label>
                      <div className="grid grid-cols-2 gap-2 h-40 overflow-y-auto border p-2 rounded">
                        {AMENITIES_LIST.map(am => {
                           const isChecked = (formData.amenities || "").includes(am);
                           return (
                             <label key={am} className="flex items-center gap-2 text-sm cursor-pointer">
                               <input 
                                 type="checkbox" 
                                 checked={isChecked} 
                                 onChange={() => handleAmenityChange(am)} 
                               />
                               {am}
                             </label>
                           );
                        })}
                      </div>
                   </div>
                </div>
              </div>

              <div className="pt-4">
                <button
                  onClick={handlePredict}
                  disabled={loading}
                  className="airbnb-btn disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? "Calculating..." : "Predict Occupancy"}
                </button>
              </div>
            </div>
          </section>
        </div>

        {/* Right Col: Prediction Card (Sticky) */}
        <div className="relative">
          <div className="sticky top-28 space-y-6">
            <div className="card shadow-xl p-6">
              <div className="flex justify-between items-start mb-4">
                <div className="flex flex-col">
                  <span className="text-xl font-bold">Prediction Result</span>
                  <span className="text-gray-500 text-sm">
                    Based on current inputs
                  </span>
                </div>
                <motion.div
                  animate={{ rotate: loading ? 360 : 0 }}
                  transition={{ repeat: Infinity, duration: 1 }}
                >
                  {loading && (
                    <div className="w-6 h-6 border-2 border-[--airbnb-red] border-t-transparent rounded-full" />
                  )}
                </motion.div>
              </div>

              <div className="border-t pt-4 min-h-[100px] flex items-center justify-center">
                {prediction ? (
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="text-center"
                  >
                    <div className="text-sm text-gray-500 uppercase tracking-wider font-bold mb-1">
                      Occupancy Category
                    </div>
                    <div
                      className={`text-4xl font-extrabold ${
                        COLORS_MAP[prediction] || "text-[--airbnb-red]"
                      }`}
                    >
                      {LABELS_MAP[prediction] || prediction} Occupancy
                    </div>
                  </motion.div>
                ) : error ? (
                  <div className="text-red-500 font-medium text-center">
                    {error}
                  </div>
                ) : (
                  <div className="text-gray-400 text-center">
                    Fill details and click Predict
                  </div>
                )}
              </div>
            </div>

            {/* Map Placeholder or Visual */}
            <div className="card h-64 bg-gray-100 flex items-center justify-center text-gray-400">
              <div className="text-center">
                <MapPin className="w-8 h-8 mx-auto mb-2" />
                <span>Map Visualization</span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
